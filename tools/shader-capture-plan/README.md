# Shader capture planning

Capture desktop shaders first by running an application that forwards Bevy's `shader_capture` feature:

```sh
BEVY_SHADER_CAPTURE_DIR=shader-captures cargo run -p your-application --features shader-capture
```

Then create a deterministic Deko3D offline plan and Rust index skeleton:

```sh
cargo run -p shader-capture-plan -- shader-captures shader-plan
```

This writes `shader-plan/manifest.json` and `shader-plan/shader_index.rs`. Source and requirement hashes are stable; no host paths or pipeline cache IDs are recorded.

To plug in a compiler, set an executable and an argument template. The tool replaces `{input}`, `{output}`, `{stage}`, `{entry_point}`, and `{defs_json}` in each argument after splitting the template on whitespace.

```sh
BEVY_DEKO3D_SHADER_COMPILER=/path/to/compiler \
BEVY_DEKO3D_SHADER_COMPILER_ARGS='--wgsl {input} --dksh {output} --stage {stage} --entry {entry_point} --defs-json {defs_json}' \
cargo run -p shader-capture-plan -- shader-captures shader-plan
```

Compiler output must start with the `DKSH` magic. Its SHA-256 is recorded in the artifact manifest. Recheck an existing artifact directory without invoking a compiler with:

```sh
cargo run -p shader-capture-plan -- shader-captures shader-plan --validate-artifacts
```
