## Executive summary (read this first)

Team 188's Development participant combines joint numerical forecasts with cited House reasoning.
It is a MindTS-inspired API adaptation, not the original anomaly detector or its neural weights.
The release workflow builds a public linux/amd64 image; team proof is packed locally and privately.

## 快速开始

1. 将源码上传到 `BrynhildF/agenthon-t2-Tipipali` 仓库的 `main` 分支。
   如果使用导出的源码 ZIP，先解压，再上传里面的文件；不要把 ZIP 当作源码上传。
   `.github/workflows/mindts-dev-image.yml` 也必须上传，保留目录结构。
2. GitHub → Actions → **MindTS Development image (team 188)** → Run workflow。
   不需要配置 Team Key。工作流不上传 CodaBench，也不消耗比赛提交次数。
3. 首次发布后，将 GHCR package 设置为 **Public**。工作流最后一步验证匿名拉取。
   如果此前因私有可见性失败，设为 Public 后重新运行。
4. 下载 `team188-development-draft` artifact，在 Linux 或 Docker 的交互终端中运行官方打包。
   Team Key 只在隐藏输入提示中输入。生成的上传文件是 `submission.zip`。
5. 使用你们固定的 CodaBench 账号，手动上传到 **Forecasting Development**。
   不要上传到 Final，不要上传源码 ZIP 或镜像文件。

详见 [完整实施与提交说明](docs/MINDTS-INTEGRATION.md)。

本地测试和离线准入检查不代表预测效果提升。镜像、真实 House 模型及 Linux 完整 smoke
只有实际运行成功后才算通过。不能在尚未构建镜像时伪造 digest 或可上传的比赛包。
