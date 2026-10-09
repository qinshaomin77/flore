# Strategy naming / 策略命名

| Strategy | Configuration / 配置 | Role / 角色 |
|---|---|---|
| Gap-actuated | `grid36/calibration/gap_actuated_calibration.yaml`, `kunshan/baseline/gap_actuated.yaml` | Conventional baseline; Grid36 file is for calibration / 常规基线；Grid36 此文件为标定用途 |
| Max-Pressure | Shared network `flore.yaml` with `--controller max_pressure` | Pressure-based baseline / 压力基线 |
| MaxPressure-TW | Shared network `flore.yaml` with `--controller truck_weighted_max_pressure` | Truck-priority baseline / 货车优先基线 |
| PressLight | `presslight/presslight_grid36.yaml`, `presslight/presslight_kunshan.yaml` | RL baseline / 强化学习基线 |
| FLORE-TS | `grid36/objective/flore_ts.yaml`, `kunshan/evaluation/flore_ts.yaml` | State ablation / 状态消融 |
| FLORE-TO | `grid36/objective/flore_to.yaml`, `kunshan/spatial_scale/flore_to_w*.yaml` | Objective ablation / 目标消融 |
| FLORE | `grid36/flore.yaml`, `kunshan/flore.yaml`, `kunshan/spatial_scale/flore_w*.yaml` | Proposed method / 提出方法 |

Filenames use lowercase snake_case; comments and task names use Table 3 spelling. / 文件名采用小写下划线，注释和任务名采用表 3 名称。

Max-Pressure and MaxPressure-TW share network configuration; their `ddqn` fields are not used to train a policy. Gap-actuated calibration files collect measurements rather than train an RL policy. / 两种压力控制器共用路网配置，不使用其中 ddqn 参数训练；标定配置仅采集数据。

NOx-only, NOx-dominant, reward and sensitivity configurations are supplemental FLORE variants, not additional main strategies. / NOx-only、NOx-dominant、奖励和敏感性配置是 FLORE 扩展实验，不作为额外主策略。

Existing model filenames are retained for checkpoint compatibility. Threshold files remain organized by network and scale. / 模型文件名保留以兼容已有权重；阈值仍按路网和尺度组织。
