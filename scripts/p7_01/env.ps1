# Spec P7-01: environment for the generation runs (dot-source: . scripts/p7_01/env.ps1).
# The price table must list every model called. The candidates get the project's deliberately
# high estimate (2 CNY in, 8 CNY out per million tokens), the same as deepseek-v4-flash and the
# judge, so CNY per question compares token use, not real prices (the gateway lists none).
$e = '{"input_cny_per_m":2.0,"output_cny_per_m":8.0,"estimated":true}'
$env:MODEL_PRICES = '{"qwen/text-embedding-v4":{"input_cny_per_m":0.5},' +
  '"deepseek-v4-flash":' + $e + ',"qwen/qwen3.6-flash":' + $e +
  ',"google/gemini-3.5-flash-lite":' + $e + ',"minimax/minimax-m3.1-flash":' + $e +
  ',"z-ai/glm-5.3-flash":' + $e + '}'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
