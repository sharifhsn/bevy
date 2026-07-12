//! Deterministically plans `Deko3D` offline shader compilation from Bevy captures.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use core::result::Result as CoreResult;
use naga::ShaderStage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

const CAPTURE_MANIFEST_VERSION: u32 = 1;
const PLAN_MANIFEST_VERSION: u32 = 1;
const COMPILER_ENV: &str = "BEVY_DEKO3D_SHADER_COMPILER";
const COMPILER_ARGS_ENV: &str = "BEVY_DEKO3D_SHADER_COMPILER_ARGS";

type Result<T> = CoreResult<T, String>;

#[derive(Deserialize, Serialize)]
struct CaptureManifest {
    version: u32,
    modules: BTreeMap<String, CapturedModule>,
}

#[derive(Deserialize, Serialize)]
struct CapturedModule {
    source: String,
    entry_points: Vec<EntryPoint>,
    uses: Vec<CaptureUse>,
}

#[derive(Clone, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct EntryPoint {
    name: String,
    stage: String,
}

#[derive(Deserialize, Serialize)]
struct CaptureUse {
    pipeline: String,
    pipeline_label: Option<String>,
    entry_point: Option<String>,
    shader_defs: Vec<ShaderDef>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct ShaderDef {
    name: String,
    kind: String,
    value: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct Requirement {
    key: String,
    source_hash: String,
    source: String,
    stage: String,
    entry_point: String,
    shader_defs: Vec<ShaderDef>,
    pipeline_label: Option<String>,
}

#[derive(Debug, Serialize)]
struct Artifact {
    key: String,
    path: String,
    sha256: Option<String>,
}

#[derive(Debug, Serialize)]
struct PlanManifest {
    version: u32,
    requirements: Vec<Requirement>,
    artifacts: Vec<Artifact>,
}

struct Compiler {
    executable: PathBuf,
    arguments: Vec<String>,
}

#[expect(
    clippy::print_stderr,
    reason = "command-line failures belong on stderr"
)]
fn main() {
    if let Err(error) = run() {
        eprintln!("shader-capture-plan: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let (capture_dir, output_dir, validate_artifacts) = parse_arguments()?;
    let compiler = compiler_from_environment()?;
    let mut plan = build_plan(&capture_dir)?;
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("could not create {}: {error}", output_dir.display()))?;
    if compiler.is_some() {
        let artifact_dir = output_dir.join("artifacts");
        fs::create_dir_all(&artifact_dir)
            .map_err(|error| format!("could not create {}: {error}", artifact_dir.display()))?;
    }

    for requirement in &plan.requirements {
        let relative_path = format!("artifacts/{}.dksh", requirement.key);
        let artifact_path = output_dir.join(&relative_path);
        let sha256 = if compiler.is_some() || validate_artifacts {
            if let Some(compiler) = &compiler {
                invoke_compiler(compiler, &capture_dir, requirement, &artifact_path)?;
            }
            Some(validate_dksh(&artifact_path)?)
        } else {
            None
        };
        plan.artifacts.push(Artifact {
            key: requirement.key.clone(),
            path: relative_path,
            sha256,
        });
    }

    write_plan(&output_dir, &plan)?;
    write_rust_index(&output_dir, &plan)?;
    Ok(())
}

fn parse_arguments() -> Result<(PathBuf, PathBuf, bool)> {
    let mut arguments = env::args_os().skip(1);
    let capture_dir = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    let output_dir = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    let validate_artifacts = match arguments.next() {
        None => false,
        Some(flag) if flag == "--validate-artifacts" => true,
        Some(_) => return Err(usage()),
    };
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((capture_dir, output_dir, validate_artifacts))
}

fn usage() -> String {
    "usage: shader-capture-plan <capture-dir> <output-dir> [--validate-artifacts]".to_string()
}

fn compiler_from_environment() -> Result<Option<Compiler>> {
    match (env::var_os(COMPILER_ENV), env::var(COMPILER_ARGS_ENV).ok()) {
        (None, None) => Ok(None),
        (Some(executable), Some(arguments)) => Ok(Some(Compiler {
            executable: PathBuf::from(executable),
            arguments: arguments
                .split_ascii_whitespace()
                .map(str::to_string)
                .collect(),
        })),
        _ => Err(format!(
            "set both {COMPILER_ENV} and {COMPILER_ARGS_ENV}, or neither"
        )),
    }
}

fn build_plan(capture_dir: &Path) -> Result<PlanManifest> {
    let manifest_path = capture_dir.join("manifest.json");
    let manifest = fs::read(&manifest_path)
        .map_err(|error| format!("could not read {}: {error}", manifest_path.display()))?;
    let manifest: CaptureManifest = serde_json::from_slice(&manifest)
        .map_err(|error| format!("could not parse {}: {error}", manifest_path.display()))?;
    if manifest.version != CAPTURE_MANIFEST_VERSION {
        return Err(format!(
            "unsupported capture manifest version {}",
            manifest.version
        ));
    }

    let mut requirements = BTreeSet::new();
    for (source_hash, module) in manifest.modules {
        validate_hash(&source_hash, "source hash")?;
        let source_path = capture_source_path(capture_dir, &module.source)?;
        let source = fs::read_to_string(&source_path)
            .map_err(|error| format!("could not read {}: {error}", source_path.display()))?;
        if sha256(source.as_bytes()) != source_hash {
            return Err(format!("source hash does not match {}", module.source));
        }
        let actual_entry_points = wgsl_entry_points(&source, &module.source)?;
        let declared_entry_points = module.entry_points.into_iter().collect::<BTreeSet<_>>();
        if declared_entry_points != actual_entry_points {
            return Err(format!("entry points do not match {}", module.source));
        }

        for capture_use in module.uses {
            let stage = pipeline_stage(&capture_use.pipeline)?;
            let shader_defs = canonical_shader_defs(capture_use.shader_defs)?;
            let entry_point =
                resolve_entry_point(&actual_entry_points, &stage, capture_use.entry_point)?;
            let mut requirement = Requirement {
                key: String::new(),
                source_hash: source_hash.clone(),
                source: module.source.clone(),
                stage,
                entry_point,
                shader_defs,
                pipeline_label: capture_use.pipeline_label,
            };
            requirement.key = requirement_hash(&requirement)?;
            if !requirements.insert(requirement) {
                return Err(format!("duplicate capture use for source {source_hash}"));
            }
        }
    }

    Ok(PlanManifest {
        version: PLAN_MANIFEST_VERSION,
        requirements: requirements.into_iter().collect(),
        artifacts: Vec::new(),
    })
}

fn capture_source_path(capture_dir: &Path, source: &str) -> Result<PathBuf> {
    let path = Path::new(source);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("capture source path is not relative: {source}"));
    }
    Ok(capture_dir.join(path))
}

fn wgsl_entry_points(source: &str, source_name: &str) -> Result<BTreeSet<EntryPoint>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| format!("could not parse {source_name} as WGSL: {error}"))?;
    Ok(module
        .entry_points
        .into_iter()
        .map(|entry_point| EntryPoint {
            name: entry_point.name,
            stage: stage_name(entry_point.stage).to_string(),
        })
        .collect())
}

fn pipeline_stage(pipeline: &str) -> Result<String> {
    let stage = match pipeline {
        "render_vertex" => "vertex",
        "render_fragment" => "fragment",
        "compute" => "compute",
        _ => return Err(format!("unsupported captured pipeline kind {pipeline}")),
    };
    Ok(stage.to_string())
}

fn resolve_entry_point(
    entry_points: &BTreeSet<EntryPoint>,
    stage: &str,
    requested: Option<String>,
) -> Result<String> {
    if let Some(requested) = requested {
        if entry_points.contains(&EntryPoint {
            name: requested.clone(),
            stage: stage.to_string(),
        }) {
            return Ok(requested);
        }
        return Err(format!("missing {stage} entry point {requested}"));
    }

    let candidates = entry_points
        .iter()
        .filter(|entry_point| entry_point.stage == stage)
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [entry_point] => Ok(entry_point.name.clone()),
        [] => Err(format!("missing {stage} entry point")),
        _ => Err(format!("ambiguous {stage} entry point")),
    }
}

fn canonical_shader_defs(shader_defs: Vec<ShaderDef>) -> Result<Vec<ShaderDef>> {
    let mut previous_name = None;
    for shader_def in &shader_defs {
        if previous_name.is_some_and(|name| name >= &shader_def.name) {
            return Err(format!(
                "shader definitions are not strictly sorted at {}",
                shader_def.name
            ));
        }
        match shader_def.kind.as_str() {
            "bool" if matches!(shader_def.value.as_str(), "true" | "false") => {}
            "int" if shader_def.value.parse::<i32>().is_ok() => {}
            "uint" if shader_def.value.parse::<u32>().is_ok() => {}
            _ => return Err(format!("invalid shader definition {}", shader_def.name)),
        }
        previous_name = Some(&shader_def.name);
    }
    Ok(shader_defs)
}

fn requirement_hash(requirement: &Requirement) -> Result<String> {
    #[derive(Serialize)]
    struct RequirementIdentity<'a> {
        source_hash: &'a str,
        stage: &'a str,
        entry_point: &'a str,
        shader_defs: &'a [ShaderDef],
        pipeline_label: &'a Option<String>,
    }
    let bytes = serde_json::to_vec(&RequirementIdentity {
        source_hash: &requirement.source_hash,
        stage: &requirement.stage,
        entry_point: &requirement.entry_point,
        shader_defs: &requirement.shader_defs,
        pipeline_label: &requirement.pipeline_label,
    })
    .map_err(|error| format!("could not serialize requirement: {error}"))?;
    Ok(sha256(&bytes))
}

fn invoke_compiler(
    compiler: &Compiler,
    capture_dir: &Path,
    requirement: &Requirement,
    output: &Path,
) -> Result<()> {
    let input = capture_source_path(capture_dir, &requirement.source)?;
    let defs_json = serde_json::to_string(&requirement.shader_defs)
        .map_err(|error| format!("could not serialize shader definitions: {error}"))?;
    match fs::remove_file(output) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("could not remove {}: {error}", output.display())),
    }
    let mut command = Command::new(&compiler.executable);
    command.args(compiler.arguments.iter().map(|argument| {
        argument
            .replace("{input}", &input.to_string_lossy())
            .replace("{output}", &output.to_string_lossy())
            .replace("{stage}", &requirement.stage)
            .replace("{entry_point}", &requirement.entry_point)
            .replace("{defs_json}", &defs_json)
    }));
    let status = command.status().map_err(|error| {
        format!(
            "could not run compiler {}: {error}",
            compiler.executable.display()
        )
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "compiler failed for {} with {status}",
            requirement.key
        ))
    }
}

fn validate_dksh(path: &Path) -> Result<String> {
    let bytes =
        fs::read(path).map_err(|error| format!("missing DKSH {}: {error}", path.display()))?;
    if !bytes.starts_with(b"DKSH") {
        return Err(format!("corrupt DKSH {}: invalid magic", path.display()));
    }
    Ok(sha256(&bytes))
}

fn write_plan(output_dir: &Path, plan: &PlanManifest) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(plan)
        .map_err(|error| format!("could not serialize plan: {error}"))?;
    fs::write(output_dir.join("manifest.json"), bytes)
        .map_err(|error| format!("could not write plan: {error}"))
}

fn write_rust_index(output_dir: &Path, plan: &PlanManifest) -> Result<()> {
    let mut index = String::from(
        "pub struct DekoShaderArtifact {\n    pub key: &'static str,\n    pub source_hash: &'static str,\n    pub stage: &'static str,\n    pub entry_point: &'static str,\n    pub shader_defs_json: &'static str,\n    pub pipeline_label: Option<&'static str>,\n    pub dksh_path: Option<&'static str>,\n}\n\npub static DEKO_SHADER_ARTIFACTS: &[DekoShaderArtifact] = &[\n",
    );
    let artifacts = plan
        .artifacts
        .iter()
        .filter(|artifact| artifact.sha256.is_some())
        .map(|artifact| (artifact.key.as_str(), artifact.path.as_str()))
        .collect::<BTreeMap<_, _>>();
    for requirement in &plan.requirements {
        let shader_defs = serde_json::to_string(&requirement.shader_defs)
            .map_err(|error| format!("could not serialize shader definitions: {error}"))?;
        let artifact = artifacts.get(requirement.key.as_str()).copied();
        index.push_str(&format!(
            "    DekoShaderArtifact {{ key: {:?}, source_hash: {:?}, stage: {:?}, entry_point: {:?}, shader_defs_json: {:?}, pipeline_label: {}, dksh_path: {} }},\n",
            requirement.key,
            requirement.source_hash,
            requirement.stage,
            requirement.entry_point,
            shader_defs,
            option_literal(requirement.pipeline_label.as_deref()),
            option_literal(artifact),
        ));
    }
    index.push_str("];\n");
    fs::write(output_dir.join("shader_index.rs"), index)
        .map_err(|error| format!("could not write Rust index: {error}"))
}

fn option_literal(value: Option<&str>) -> String {
    value.map_or_else(|| "None".to_string(), |value| format!("Some({value:?})"))
}

fn validate_hash(hash: &str, description: &str) -> Result<()> {
    if hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("invalid {description} {hash}"))
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn stage_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Task => "task",
        ShaderStage::Mesh => "mesh",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
        ShaderStage::RayGeneration => "ray_generation",
        ShaderStage::Miss => "ray_miss",
        ShaderStage::AnyHit => "ray_any_hit",
        ShaderStage::ClosestHit => "ray_closest_hit",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::process;

    fn test_directory(name: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = env::temp_dir().join(format!(
            "shader-capture-plan-{name}-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(directory.join("sources")).unwrap();
        directory
    }

    fn shader_defs() -> Vec<ShaderDef> {
        vec![ShaderDef {
            name: "A".to_string(),
            kind: "bool".to_string(),
            value: "true".to_string(),
        }]
    }

    fn write_capture(directory: &Path, uses: Vec<CaptureUse>) {
        let source = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }\n";
        let source_hash = sha256(source.as_bytes());
        fs::write(
            directory.join(format!("sources/{source_hash}.wgsl")),
            source,
        )
        .unwrap();
        let mut modules = BTreeMap::new();
        modules.insert(
            source_hash,
            CapturedModule {
                source: format!("sources/{}.wgsl", sha256(source.as_bytes())),
                entry_points: vec![EntryPoint {
                    name: "vs".to_string(),
                    stage: "vertex".to_string(),
                }],
                uses,
            },
        );
        let manifest = CaptureManifest {
            version: CAPTURE_MANIFEST_VERSION,
            modules,
        };
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn vertex_use() -> CaptureUse {
        CaptureUse {
            pipeline: "render_vertex".to_string(),
            pipeline_label: Some("pipeline".to_string()),
            entry_point: None,
            shader_defs: shader_defs(),
        }
    }

    #[test]
    fn plan_is_deterministic() {
        let first = test_directory("deterministic-first");
        let second = test_directory("deterministic-second");
        write_capture(&first, vec![vertex_use()]);
        write_capture(&second, vec![vertex_use()]);
        let first_plan = build_plan(&first).unwrap();
        let second_plan = build_plan(&second).unwrap();
        assert_eq!(
            serde_json::to_vec_pretty(&first_plan).unwrap(),
            serde_json::to_vec_pretty(&second_plan).unwrap()
        );
        let output = test_directory("deterministic-output");
        write_rust_index(&output, &first_plan).unwrap();
        let first_index = fs::read(output.join("shader_index.rs")).unwrap();
        write_rust_index(&output, &second_plan).unwrap();
        assert_eq!(
            first_index,
            fs::read(output.join("shader_index.rs")).unwrap()
        );
        fs::remove_dir_all(first).unwrap();
        fs::remove_dir_all(second).unwrap();
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn rejects_duplicate_and_conflicting_capture_uses() {
        let duplicate = test_directory("duplicate");
        write_capture(&duplicate, vec![vertex_use(), vertex_use()]);
        assert!(build_plan(&duplicate).unwrap_err().contains("duplicate"));

        let conflicting = test_directory("conflicting");
        let mut conflicting_use = vertex_use();
        conflicting_use.shader_defs.push(ShaderDef {
            name: "A".to_string(),
            kind: "uint".to_string(),
            value: "1".to_string(),
        });
        write_capture(&conflicting, vec![conflicting_use]);
        assert!(build_plan(&conflicting)
            .unwrap_err()
            .contains("not strictly sorted"));
        fs::remove_dir_all(duplicate).unwrap();
        fs::remove_dir_all(conflicting).unwrap();
    }

    #[test]
    fn rejects_missing_and_corrupt_dksh() {
        let directory = test_directory("dksh");
        let artifact = directory.join("artifacts/shader.dksh");
        assert!(validate_dksh(&artifact)
            .unwrap_err()
            .contains("missing DKSH"));
        fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        fs::write(&artifact, b"not a shader").unwrap();
        assert!(validate_dksh(&artifact)
            .unwrap_err()
            .contains("invalid magic"));
        fs::write(&artifact, b"DKSH payload").unwrap();
        assert_eq!(validate_dksh(&artifact).unwrap(), sha256(b"DKSH payload"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_missing_and_ambiguous_entry_points() {
        let missing = test_directory("missing-entry");
        let mut missing_use = vertex_use();
        missing_use.entry_point = Some("missing".to_string());
        write_capture(&missing, vec![missing_use]);
        assert!(build_plan(&missing)
            .unwrap_err()
            .contains("missing vertex entry point"));

        let ambiguous = test_directory("ambiguous-entry");
        let source = "@vertex fn first() -> @builtin(position) vec4f { return vec4f(); }\n@vertex fn second() -> @builtin(position) vec4f { return vec4f(); }\n";
        let source_hash = sha256(source.as_bytes());
        fs::write(
            ambiguous.join(format!("sources/{source_hash}.wgsl")),
            source,
        )
        .unwrap();
        let mut modules = BTreeMap::new();
        modules.insert(
            source_hash.clone(),
            CapturedModule {
                source: format!("sources/{source_hash}.wgsl"),
                entry_points: vec![
                    EntryPoint {
                        name: "first".to_string(),
                        stage: "vertex".to_string(),
                    },
                    EntryPoint {
                        name: "second".to_string(),
                        stage: "vertex".to_string(),
                    },
                ],
                uses: vec![vertex_use()],
            },
        );
        fs::write(
            ambiguous.join("manifest.json"),
            serde_json::to_vec_pretty(&CaptureManifest {
                version: CAPTURE_MANIFEST_VERSION,
                modules,
            })
            .unwrap(),
        )
        .unwrap();
        assert!(build_plan(&ambiguous)
            .unwrap_err()
            .contains("ambiguous vertex entry point"));
        fs::remove_dir_all(missing).unwrap();
        fs::remove_dir_all(ambiguous).unwrap();
    }
}
