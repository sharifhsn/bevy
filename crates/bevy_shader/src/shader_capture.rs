use super::{ShaderCaptureContext, ShaderDefVal};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    vec::Vec,
};
use naga::ShaderStage;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

const MANIFEST_VERSION: u32 = 1;
const OUTPUT_DIR_ENV: &str = "BEVY_SHADER_CAPTURE_DIR";

#[derive(Serialize)]
struct Manifest {
    version: u32,
    modules: BTreeMap<String, CapturedModule>,
}

#[derive(Serialize)]
struct CapturedModule {
    source: String,
    entry_points: Vec<EntryPoint>,
    uses: BTreeSet<CaptureUse>,
}

#[derive(Serialize)]
struct EntryPoint {
    name: String,
    stage: &'static str,
}

#[derive(Serialize, Ord, PartialOrd, Eq, PartialEq)]
struct CaptureUse {
    pipeline: &'static str,
    pipeline_label: Option<String>,
    entry_point: Option<String>,
    shader_defs: Vec<ShaderDef>,
}

#[derive(Serialize, Ord, PartialOrd, Eq, PartialEq)]
struct ShaderDef {
    name: String,
    kind: &'static str,
    value: String,
}

pub(super) struct ShaderCapture {
    output_dir: PathBuf,
    manifest: Manifest,
}

impl ShaderCapture {
    pub(super) fn from_environment() -> io::Result<Self> {
        let output_dir = env::var_os(OUTPUT_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("shader-captures"));
        fs::create_dir_all(output_dir.join("sources"))?;
        Ok(Self {
            output_dir,
            manifest: Manifest {
                version: MANIFEST_VERSION,
                modules: BTreeMap::new(),
            },
        })
    }

    pub(super) fn capture(
        &mut self,
        wgsl: String,
        shader_defs: Vec<ShaderDefVal>,
        entry_points: Vec<(String, ShaderStage)>,
        context: &ShaderCaptureContext,
    ) -> io::Result<()> {
        let key = sha256(&wgsl);
        let source = format!("sources/{key}.wgsl");
        let entry_points = entry_points
            .into_iter()
            .map(|(name, stage)| EntryPoint {
                name,
                stage: stage_name(stage),
            })
            .collect();
        let captured_module =
            self.manifest
                .modules
                .entry(key.clone())
                .or_insert_with(|| CapturedModule {
                    source: source.clone(),
                    entry_points,
                    uses: BTreeSet::new(),
                });
        captured_module.uses.insert(CaptureUse {
            pipeline: context.pipeline.as_str(),
            pipeline_label: context.pipeline_label.clone(),
            entry_point: context.entry_point.clone(),
            shader_defs: canonical_shader_defs(shader_defs),
        });

        write_if_missing(&self.output_dir.join(source), wgsl.as_bytes())?;
        self.write_manifest()
    }

    fn write_manifest(&self) -> io::Result<()> {
        let manifest = serde_json::to_vec_pretty(&self.manifest).map_err(io::Error::other)?;
        fs::write(self.output_dir.join("manifest.json"), manifest)
    }
}

fn sha256(wgsl: &str) -> String {
    format!("{:x}", Sha256::digest(wgsl.as_bytes()))
}

fn canonical_shader_defs(shader_defs: Vec<ShaderDefVal>) -> Vec<ShaderDef> {
    let mut defs = BTreeMap::new();
    for shader_def in shader_defs {
        let (name, kind, value) = match shader_def {
            ShaderDefVal::Bool(name, value) => (name, "bool", value.to_string()),
            ShaderDefVal::Int(name, value) => (name, "int", value.to_string()),
            ShaderDefVal::UInt(name, value) => (name, "uint", value.to_string()),
        };
        defs.insert(name, (kind, value));
    }
    defs.into_iter()
        .map(|(name, (kind, value))| ShaderDef { name, kind, value })
        .collect()
}

fn stage_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
        ShaderStage::Task => "task",
        ShaderStage::Mesh => "mesh",
        ShaderStage::RayGeneration => "ray_generation",
        ShaderStage::Miss => "ray_miss",
        ShaderStage::AnyHit => "ray_any_hit",
        ShaderStage::ClosestHit => "ray_closest_hit",
    }
}

fn write_if_missing(path: &Path, bytes: &[u8]) -> io::Result<()> {
    match fs::read(path) {
        Ok(existing) if existing == bytes => Ok(()),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("shader capture key collision at {}", path.display()),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => fs::write(path, bytes),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShaderCapturePipeline;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::process;

    fn test_output_dir(name: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let output_dir = env::temp_dir().join(format!(
            "bevy-shader-capture-test-{name}-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&output_dir);
        output_dir
    }

    #[test]
    fn canonical_shader_defs_are_sorted_and_last_value_wins() {
        let defs = canonical_shader_defs(vec![
            ShaderDefVal::UInt("B".to_string(), 1),
            ShaderDefVal::Bool("A".to_string(), true),
            ShaderDefVal::Int("B".to_string(), -2),
        ]);
        assert_eq!(
            defs.into_iter()
                .map(|def| (def.name, def.value))
                .collect::<Vec<_>>(),
            vec![
                ("A".to_string(), "true".to_string()),
                ("B".to_string(), "-2".to_string()),
            ]
        );
    }

    #[test]
    fn captures_deduplicate_source_and_sort_uses() {
        let output_dir = test_output_dir("deduplicate");
        fs::create_dir_all(output_dir.join("sources")).unwrap();
        let mut capture = ShaderCapture {
            output_dir: output_dir.clone(),
            manifest: Manifest {
                version: MANIFEST_VERSION,
                modules: BTreeMap::new(),
            },
        };
        let source = "@compute @workgroup_size(1) fn main() {}\n".to_string();
        let first = ShaderCaptureContext::new(ShaderCapturePipeline::Compute, None, None);
        let second = ShaderCaptureContext::new(
            ShaderCapturePipeline::Compute,
            Some("second".to_string()),
            Some("main".to_string()),
        );
        capture
            .capture(
                source.clone(),
                vec![ShaderDefVal::Bool("B".to_string(), true)],
                vec![("main".to_string(), ShaderStage::Compute)],
                &second,
            )
            .unwrap();
        capture
            .capture(
                source,
                vec![ShaderDefVal::Bool("A".to_string(), true)],
                vec![("main".to_string(), ShaderStage::Compute)],
                &first,
            )
            .unwrap();

        assert_eq!(capture.manifest.modules.len(), 1);
        let module = capture.manifest.modules.values().next().unwrap();
        assert_eq!(module.uses.len(), 2);
        assert_eq!(fs::read_dir(output_dir.join("sources")).unwrap().count(), 1);
        let manifest = fs::read_to_string(output_dir.join("manifest.json")).unwrap();
        assert!(
            manifest.find("\"pipeline_label\": null").unwrap()
                < manifest.find("\"pipeline_label\": \"second\"").unwrap()
        );
        fs::remove_dir_all(output_dir).unwrap();
    }

    #[test]
    fn manifest_is_deterministic_across_capture_order() {
        let base = test_output_dir("determinism");
        let first_dir = base.join("first");
        let second_dir = base.join("second");
        let source = "@compute @workgroup_size(1) fn main() {}\n".to_string();
        let contexts = [
            ShaderCaptureContext::new(ShaderCapturePipeline::Compute, None, None),
            ShaderCaptureContext::new(
                ShaderCapturePipeline::Compute,
                Some("second".to_string()),
                Some("main".to_string()),
            ),
        ];

        for (output_dir, order) in [(&first_dir, [0, 1]), (&second_dir, [1, 0])] {
            fs::create_dir_all(output_dir.join("sources")).unwrap();
            let mut capture = ShaderCapture {
                output_dir: output_dir.clone(),
                manifest: Manifest {
                    version: MANIFEST_VERSION,
                    modules: BTreeMap::new(),
                },
            };
            for index in order {
                capture
                    .capture(
                        source.clone(),
                        vec![ShaderDefVal::Bool(
                            (char::from(b'A' + index as u8)).to_string(),
                            true,
                        )],
                        vec![("main".to_string(), ShaderStage::Compute)],
                        &contexts[index],
                    )
                    .unwrap();
            }
        }

        assert_eq!(
            fs::read(first_dir.join("manifest.json")).unwrap(),
            fs::read(second_dir.join("manifest.json")).unwrap()
        );
        fs::remove_dir_all(base).unwrap();
    }
}
