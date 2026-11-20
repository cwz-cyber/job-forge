# OpenHaven

OpenHaven 是一个面向求职者的材料加工智能体原型。它只生成由用户人工核对、复制和提交的文本，不提供网页访问、表单填写、附件上传或自动投递能力。

这个仓库目前包含：

- 六类求职工作流的统一编排层；
- 带来源 ID 的候选人素材库；
- 技术赛道与体制赛道两套提示词；
- 字数、数字、敏感资历词、素材引用和禁用动作校验；
- 同一招聘场次的上下文快照和投递台账；
- 无第三方依赖的 OpenAI-compatible 模型适配器；
- 可离线运行的测试与 dry-run 模式。

## 快速开始

要求 Python 3.11 或更高版本。

```bash
python3 -m unittest discover -s tests -v
```

查看智能体将发送给模型的请求包（不会联网）：

```bash
PYTHONPATH=src python3 -m openhaven \
  --profile data/candidate_profile.example.json \
  --request examples/jd_analysis.json \
  --dry-run
```

接入兼容 Chat Completions 的模型服务：

```bash
export OPENHAVEN_API_KEY='your-key'
PYTHONPATH=src python3 -m openhaven \
  --profile data/candidate_profile.example.json \
  --request examples/jd_analysis.json \
  --endpoint https://your-provider.example/v1/chat/completions \
  --model your-model
```

命令只读取本地 JSON、请求指定的大模型接口并输出文本结果。代码中没有浏览器自动化、招聘网站接口或文件上传实现。

## 请求格式

```json
{
  "workflow": "jd_analysis",
  "style": "technical",
  "application_id": "company-role-2027",
  "payload": {
    "jd_text": "粘贴岗位描述"
  }
}
```

支持的 `workflow`：

- `jd_analysis`
- `form_answers`
- `resume_rewrite`
- `interview_kit`
- `application_ledger`
- `cover_letter`

完整设计、边界与生产化建议见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
