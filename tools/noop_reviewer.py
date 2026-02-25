import json, sys
_ = sys.stdin.read()
print(json.dumps({
  "patch_unified_diff": "",
  "score": 0,
  "summary": "noop reviewer (no api)",
  "risks": []
}))
