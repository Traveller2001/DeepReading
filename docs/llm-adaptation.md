# 适配自己的大模型接口

DeepReading 的所有 LLM 调用都通过 `llm_client.py` 进行。要切换模型，大多数情况下只改 `.env` 就够了。

---

## 快速切换

只要你的提供商兼容 OpenAI API（DeepSeek、OpenAI、智谱、Kimi、OpenRouter、vLLM、Ollama 等），设置三个环境变量即可：

```bash
LLM_API_BASE=https://api.deepseek.com   # 换成你的 API 地址
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
```

Azure OpenAI 额外设置 `LLM_API_VERSION`：

```bash
LLM_API_BASE=https://your-resource.openai.azure.com
LLM_API_KEY=your-azure-key
LLM_API_VERSION=2024-12-01-preview
LLM_MODEL=your-deployment-name
```

---

## 两种 LLM 角色

| 角色 | 环境变量 | 要求 |
|------|----------|------|
| **Main** | `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` | **tool calling** + **streaming** |
| **Vision** | `VISION_MODEL`（共用 Main 的 API 地址和 Key） | **图片输入** |

`VISION_MODEL` 不设则自动使用 `LLM_MODEL`。如果模型不支持 vision，图表提取会降级为基于规则的方式。

### 流程中谁调了什么

| 流程步骤 | 角色 | 调用函数 | 调用方 |
|----------|------|----------|--------|
| 报告生成（tool-calling loop） | Main | `generate_stream()` | `core/llm_service.py` |
| 讨论 Q&A | Main | `generate_stream()` | `core/llm_service.py` |
| 报告润色 | Main | `generate_stream()` | `core/llm_service.py` |
| PDF 图表识别 | Vision | `generate_sync()` | `processor/vision_extractor.py` |
| 生成图表质量审核 | Vision | `generate_sync()` | `tools/figure_reviewer.py` |

如果你只想换报告生成的模型，改 `LLM_MODEL`；只想换视觉相关的模型，改 `VISION_MODEL`。两者互不影响。

### 分开配置示例

```bash
LLM_MODEL=deepseek-chat
REPORT_MODEL=deepseek-reasoner     # 报告生成单独用更强的模型
VISION_MODEL=gpt-4o-mini           # 视觉任务用便宜的模型
```

---

## 进阶：自定义 `llm_client.py`

仅当你的 API 格式不兼容 OpenAI 时才需要改。文件只暴露两个函数：

- **`generate_stream(messages, **kwargs)`** → `AsyncIterator[StreamChunk]`：异步流式，报告/讨论/润色全走这里
- **`generate_sync(text, *, messages, images, system, **kwargs)`** → `str`：同步调用，Vision 和图表审核走这里

### 关键约定

`generate_stream()` 必须 yield `StreamChunk`，业务层依赖这些字段：

```python
@dataclass
class StreamChunk:
    content: str | None = None          # 文本增量
    reasoning: str | None = None        # 推理过程（如 DeepSeek R1）
    tool_calls: list[ToolCallDelta] | None = None  # 工具调用
    finish_reason: str | None = None    # "stop" / "tool_calls" / "length"
```

只要正确填充这些字段，业务层不需要任何修改。

### 示例：适配自定义 HTTP API

```python
import httpx

async def generate_stream(messages, **kwargs):
    prompt = "\n".join(m["content"] for m in messages if isinstance(m["content"], str))

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", "https://your-api.com/chat",
            json={"prompt": prompt, "stream": True},
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                yield StreamChunk(
                    content=data.get("text"),
                    finish_reason="stop" if data.get("done") else None,
                )
```

### 模型不支持 tool calling？

报告生成会降级为直接生成（质量下降但不报错）。确保 `generate_stream()` 收到 `tools` 参数时不报错即可：

```python
async def generate_stream(messages, **kwargs):
    kwargs.pop("tools", None)  # 忽略工具定义
    # ... 正常调用
```

---

## 常见问题

**报告没有使用工具调用** — 模型不支持 function calling，系统已降级。推荐 DeepSeek、GPT-4o 等支持 tool calling 的模型。

**图表提取返回空** — `VISION_MODEL` 不支持多图输入，系统已降级为规则提取。

**报告生成中途断开** — `LLM_TIMEOUT`（默认 180s）太短，或 `MAX_REPORT_TOKENS`（默认 65536）超过模型上下文窗口。
