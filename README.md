# Ather Exploration — G0/G1/G2/G3

Dự án RL exploration dựa trên official **MiniGrid v3.1.0**. G0 có nền tảng/config/contracts;
G1 có môi trường chạy trên fixture: movement/collision, FOV, reward, memory và Gymnasium API.
G2 có generator/validator procedural, cache và development scenario bank. G3 có metrics/evaluator và hai baseline random/frontier. G3-UI có workbench tương tác để kiểm từng phase. G4 có runner/checkpoint/curriculum và learned inference; chưa chạy train theo yêu cầu người dùng, learning smoke còn chờ.

**Đọc theo thứ tự:** [G0](notes/G0_GUIDE.md) → [G1: code, ví dụ và kết quả](notes/G1_GUIDE.md) → [G2: sinh map, witness, bank và cách kiểm tra](notes/G2_GUIDE.md).
Thiết kế và phạm vi triển khai nằm trong [design](notes/design.md) / [plan](notes/plan.md).

**Muốn hiểu logic và luồng dữ liệu trước khi chạy:**

| Gate | Giải thích kiến trúc, thuật toán và sơ đồ | Guide chạy/kiểm tra |
| --- | --- | --- |
| G0 | [Nền tảng, config, contracts, schema](notes/G0_EXPLAINED.md) | [G0 guide](notes/G0_GUIDE.md) |
| G1 | [Reset/step, collisions, FOV, reward, memory](notes/G1_EXPLAINED.md) | [G1 guide](notes/G1_GUIDE.md) |
| G2 | [BSP, temporal search, seeds, cache/bank](notes/G2_EXPLAINED.md) | [G2 guide](notes/G2_GUIDE.md) |
| G3 | [Metrics, baseline và luồng evaluator](notes/G3_EXPLAINED.md) | [G3 guide](notes/G3_GUIDE.md) |
| G4 | [Encoder, runner, curriculum, checkpoint và replay](notes/G4_EXPLAINED.md) | [Tự đọc/chuẩn bị/chạy train](notes/G4_GUIDE.md) |
| G3-UI | [UI, worker, render và headless training](notes/G3_UI_EXPLAINED.md) | [UI guide](notes/G3_UI_GUIDE.md) |

Các sơ đồ được nhúng bằng ảnh SVG trong `notes/diagrams/`, hiển thị trực tiếp trong
Markdown preview và không cần extension Mermaid. Bấm “Mở sơ đồ ở kích thước đầy đủ”
để xem hình lớn; mã nguồn Mermaid được giữ trong mục thu gọn ngay dưới mỗi hình.


## Cài và kiểm tra

Chạy từ project root mới, dùng `uv` đã có trên máy:

```bash
cd /Users/vominhthinh/Workspace/ather_lab
uv sync --locked --extra cpu
source .venv/bin/activate
python -m ather_exploration --help
python -m ather_exploration config --preset medium
python -m pytest tests -q
python -m ather_exploration setup-check
python -m ather_exploration rollout --fixture collision2_poi
python -m ather_exploration generate --preset small --seed 42
python -m ather_exploration evaluate --presets small --seeds 42 --action-repeats 1
```

`uv sync` cài Python dependencies trong `.venv`; profile `cpu` bao gồm thư viện RL
và UI, dev group mặc định có pytest/build tools. Smoke dùng CPU và SDL dummy,
không mở cửa sổ, không gọi `learn()` hoặc Modal. Lần đầu import Torch có thể lâu.
JSON cuối của smoke phải có `"status": "pass"`.

Không chạy `uv sync` thiếu `--extra cpu` nếu đang cần các thư viện RL/UI: uv có thể
loại extras khỏi venv. Có thể chạy trực tiếp `.venv/bin/python` để không phụ thuộc
shell đã activate hay chưa.

## Cấu trúc

- `notes/`: đề, design, plan, research và hướng dẫn và kết quả từng gate.
- `minigrid/`: source upstream chính thức, được giữ nguyên trong G0.
- `ather_exploration/`: contracts dùng chung (`config.py`, `types.py`, `schema.py`), seeds/fixtures/setup và CLI.
- `ather_exploration/environment/`: env Gymnasium, dynamics, FOV, public state, memory, reward.
- `ather_exploration/worlds/`: generation, topology, temporal validation, scenario persistence và banks.
- `ather_exploration/agents/`: random/frontier; G4 thêm encoder và learned adapters.
- `ather_exploration/evaluation/`: episode/batch runner và metrics.
- `ather_exploration/ui/`: app, session/controller, renderer và telemetry reader.
- `ather_exploration/resources/`: nguồn duy nhất cho presets và fixtures, được đóng gói vào wheel.
- `tests/`: chỉ giữ tests của bài hiện tại.
- Kết quả từng gate được ghi trong `notes/G0_GUIDE.md`, `notes/G1_GUIDE.md`, `notes/G2_GUIDE.md`, `notes/G3_GUIDE.md`.
- `uv.lock`: lock duy nhất cho toàn bộ dependency profiles.

## Dependency profiles và giới hạn nền tảng

Python `3.11.x` (máy này đã kiểm chứng `3.11.14`). SB3/Contrib `2.9.0`,
Gymnasium `1.3.0`, Torch `2.11.0`. Toàn bộ transitive versions/hashes nằm trong locks.

- Local CPU/UI/dev: `uv sync --locked --extra cpu`.
- CPU/UI runtime không dev: `uv sync --locked --extra cpu --no-dev`.
- Linux CUDA/Modal: `uv sync --locked --extra cuda`; PyTorch CUDA 12.8 index.
  Đây là profile đã resolve, **chưa chạy trên GPU/Modal**. Kiểm chứng driver/device
  và throughput ở S18. Không sync profile này vào venv local đang dùng nếu không cần.
- `cpu` và `cuda` là extras loại trừ nhau; không dùng `--all-extras`.
- Chỉ `pygame-ce` cung cấp import `pygame`, không cài thêm distribution `pygame`.
- Distribution dự án cung cấp cả `minigrid` và `ather_exploration`; không cài thêm
  một wheel `minigrid` khác vào cùng venv.

Không lưu thêm requirements lock trùng lặp. `uv sync --locked` chọn đúng profile/index
trong `uv.lock`. Không cần export thêm lock cho công việc hiện tại.

Các cache, `.venv`, metadata và output build được bỏ qua trong Git và ẩn trong
VS Code Explorer bằng `.vscode/settings.json`. `.venv` là môi trường cần để chạy,
không phải source cần đọc/sửa. Metadata có thể được tạo lại khi cài editable.

## Nguồn và giấy phép

Source nền: [official MiniGrid v3.1.0](https://github.com/Farama-Foundation/Minigrid/tree/v3.1.0),
commit `90928729376741a41222a257911343b97103b548` của Farama Foundation.
[MIT license](LICENSE) được giữ nguyên. Local branch `ather-g0` bắt đầu từ commit này;
remote `upstream` trỏ về repository chính thức. Chưa push/publish dự án.

`minigrid/` được giữ; tests cho các môi trường khác của upstream đã được bỏ khỏi cây làm việc. Logic thực thi MiniGrid không thay đổi;
các liên kết hình trong docstring được đổi sang URL upstream sau khi bỏ demo assets.
`ather_exploration/` là phần triển khai cho bài. Source website, hình demo, logo,
workflow/template quản trị, funding và code-of-conduct upstream không còn trong
cây làm việc. Bản gốc vẫn truy cập được qua upstream commit.

[README gốc](notes/upstream/README.md), metadata packaging gốc và CITATION được lưu
ở `notes/upstream/` để tham chiếu; chúng không phải cấu hình build hiện hành.
Các thư viện khác được cài vào `.venv`, versions/hashes nằm trong `uv.lock`.
Kiểm kê license của dependencies/assets khi đóng gói native thuộc S17/S26.

## UI kiểm tra trước G4

```bash
python -m ather_exploration ui --preset small --seed 42 --agent frontier
python -m ather_exploration ui --fixture collision2_poi --agent random
```

UI tự chơi, hiển thị world/public memory; New map, Reset same map và dropdown chọn checkpoint khi dùng --checkpoint-dir.
Config/seed/replay chọn bằng CLI; metrics train xem W&B. Đọc [UI guide](notes/G3_UI_GUIDE.md).
Training hỗ trợ headless hoặc viewer local; xem [Modal guide](notes/MODAL_GUIDE.md).
Train headless; tải checkpoint xong mở viewer riêng. Dữ liệu thử cũ đã dọn; chưa chạy learning cho revision mới.

Cấu trúc package đã được gom trước G4; xem [design §4.3](notes/design.md#d04).
Ví dụ import hiện hành: `from ather_exploration.environment.env import make_env`.
CLI giữ nguyên; paths phẳng cũ không còn được cung cấp. Fingerprint source quét đệ quy;
bank/cache tạo trước refactor cần generate lại theo G2 guide.

G4: [config mẫu](ather_exploration/resources/training/development.yaml), `train-check` không huấn luyện; `train` mới bắt đầu cập nhật trọng số. Chưa có learning-quality evidence hoặc Modal job.
