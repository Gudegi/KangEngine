# RL With MimicKit

KangEngine can be used as a backend engine of
[MimicKit](https://github.com/xbpeng/MimicKit) through KangEngine's Python
package. Use the `backend_kangengine` branch of MimicKit and keep MimicKit in a
separate Python environment.

Check [Installation](getting_started/INSTALLATION.md) for the wheel's platform
and NVIDIA driver requirements.

![MimicKit running with KangEngine](../images/Mimickit_kangengine_1.png)

## MimicKit setup and run commands

1. Clone the KangEngine-enabled MimicKit fork branch.

    ```bash
    git clone -b backend_kangengine https://github.com/Gudegi/MimicKit.git
    ```

2. Create and activate a MimicKit Python environment with uv.

    ```bash
    cd MimicKit
    uv venv .venv --python 3.12
    source .venv/bin/activate
    ```

3. Install KangEngine into the active MimicKit environment.

    ```bash
    uv pip install kangengine
    ```

4. Install MimicKit dependencies.

    ```bash
    cd /path/to/MimicKit
    uv pip install -r requirements.txt
    ```

5. Run a small motion visualization test.

    ```bash
    python mimickit/run.py \
      --mode test \
      --num_envs 1 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/view_motion_humanoid_env.yaml \
      --visualize true \
      --devices cpu \
      --test_episodes 10
    ```

6. Run pretrained policy inference with KangEngine.

    ```bash
    python mimickit/run.py \
      --mode test \
      --num_envs 4 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/amp_humanoid_env.yaml \
      --agent_config data/agents/amp_humanoid_agent.yaml \
      --visualize true \
      --model_file data/models/amp_humanoid_spinkick_model.pt
    ```

7. Train an AMP policy with KangEngine.

    ```bash
    python mimickit/run.py \
      --mode train \
      --num_envs 4096 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/amp_humanoid_env.yaml \
      --agent_config data/agents/amp_humanoid_agent.yaml \
      --visualize false \
      --out_dir output/
    ```

For reference, the MimicKit KangEngine backend uses an engine config like this:

```yaml
engine_name: "kangengine"

control_mode: "pos"
control_freq: 30
sim_freq: 120
env_spacing: 5
enable_self_collisions: false
```

The `backend_kangengine` branch already includes
`data/engines/kangengine_engine.yaml`, so you usually do not need to create it
manually.
