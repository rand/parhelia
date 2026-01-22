#!/usr/bin/env python3
"""End-to-end MCP integration test."""

import json
import subprocess
import sys

def send_mcp_request(method: str, params: dict = None) -> dict:
    """Send a request to the MCP server and return the response."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
    }
    if params:
        request["params"] = params
    
    proc = subprocess.run(
        ["uv", "run", "parhelia", "mcp-server"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/Users/rand/src/parhelia"
    )
    
    if proc.returncode != 0 and not proc.stdout:
        print(f"STDERR: {proc.stderr}", file=sys.stderr)
        return {"error": proc.stderr}
    
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "stdout": proc.stdout, "stderr": proc.stderr}

def test_initialize():
    """Test MCP initialization."""
    print("Testing: initialize")
    resp = send_mcp_request("initialize")
    assert "result" in resp, f"No result: {resp}"
    assert resp["result"]["serverInfo"]["name"] == "parhelia"
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    print("  ✓ Server initializes correctly")
    return True

def test_tools_list():
    """Test tools/list returns all tools."""
    print("Testing: tools/list")
    resp = send_mcp_request("tools/list")
    assert "result" in resp, f"No result: {resp}"
    tools = resp["result"]["tools"]
    assert len(tools) >= 15, f"Expected 15+ tools, got {len(tools)}"
    
    tool_names = {t["name"] for t in tools}
    required = [
        "parhelia_containers",
        "parhelia_container_show",
        "parhelia_health",
        "parhelia_task_create",
        "parhelia_task_list",
        "parhelia_budget_status",
        "parhelia_checkpoint_list",
    ]
    for name in required:
        assert name in tool_names, f"Missing tool: {name}"
    
    print(f"  ✓ {len(tools)} tools available")
    return True

def test_health():
    """Test parhelia_health tool."""
    print("Testing: parhelia_health")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_health",
        "arguments": {}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "overall_health" in content, f"No overall_health: {content}"
    print(f"  ✓ Health: {content['overall_health']}")
    return True

def test_containers_list():
    """Test parhelia_containers tool."""
    print("Testing: parhelia_containers")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_containers",
        "arguments": {"limit": 5}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "containers" in content, f"No containers key: {content}"
    print(f"  ✓ Listed {len(content['containers'])} containers")
    return True

def test_task_list():
    """Test parhelia_task_list tool."""
    print("Testing: parhelia_task_list")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_task_list",
        "arguments": {"limit": 5}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "tasks" in content, f"No tasks key: {content}"
    print(f"  ✓ Listed {len(content['tasks'])} tasks")
    return True

def test_budget_status():
    """Test parhelia_budget_status tool."""
    print("Testing: parhelia_budget_status")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_budget_status",
        "arguments": {}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    budget = content.get("budget", content)
    assert "ceiling_usd" in budget, f"No ceiling_usd: {content}"
    print(f"  ✓ Budget: ${budget['used_usd']:.2f} / ${budget['ceiling_usd']:.2f}")
    return True

def test_budget_estimate():
    """Test parhelia_budget_estimate tool."""
    print("Testing: parhelia_budget_estimate")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_budget_estimate",
        "arguments": {
            "task_type": "test_run",
            "timeout_hours": 1
        }
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "estimated_cost_usd" in content, f"No cost estimate: {content}"
    print(f"  ✓ Estimated cost: ${content['estimated_cost_usd']:.4f}")
    return True

def test_reconciler_status():
    """Test parhelia_reconciler_status tool."""
    print("Testing: parhelia_reconciler_status")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_reconciler_status",
        "arguments": {}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "is_running" in content, f"No is_running: {content}"
    print(f"  ✓ Reconciler running: {content['is_running']}")
    return True

def test_checkpoint_list():
    """Test parhelia_checkpoint_list tool."""
    print("Testing: parhelia_checkpoint_list")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_checkpoint_list",
        "arguments": {"session_id": "test-session", "limit": 5}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "checkpoints" in content, f"No checkpoints key: {content}"
    print(f"  ✓ Listed {len(content['checkpoints'])} checkpoints")
    return True

def test_task_create():
    """Test parhelia_task_create with dispatch=false."""
    print("Testing: parhelia_task_create (no dispatch)")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_task_create",
        "arguments": {
            "prompt": "echo 'MCP integration test'",
            "dispatch": False
        }
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "task_id" in content, f"No task_id: {content}"
    assert "estimated_cost_usd" in content, f"No estimated_cost_usd: {content}"
    print(f"  ✓ Created task: {content['task_id']}")
    print(f"    Estimated: ${content['estimated_cost_usd']:.2f}")
    return True

def test_container_show_not_found():
    """Test parhelia_container_show with invalid ID."""
    print("Testing: parhelia_container_show (not found)")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_container_show",
        "arguments": {"container_id": "c-nonexistent"}
    })
    assert "result" in resp, f"No result: {resp}"
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content.get("success") is False or content.get("container") is None
    print(f"  ✓ Not found handled correctly")
    return True

def test_unknown_tool():
    """Test unknown tools return error."""
    print("Testing: unknown tool error")
    resp = send_mcp_request("tools/call", {
        "name": "parhelia_nonexistent",
        "arguments": {}
    })
    assert "error" in resp, f"Expected error: {resp}"
    print(f"  ✓ Unknown tool returns error")
    return True

def main():
    print("=" * 60)
    print("Parhelia MCP Integration E2E Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_initialize,
        test_tools_list,
        test_health,
        test_containers_list,
        test_task_list,
        test_budget_status,
        test_budget_estimate,
        test_reconciler_status,
        test_checkpoint_list,
        test_task_create,
        test_container_show_not_found,
        test_unknown_tool,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
