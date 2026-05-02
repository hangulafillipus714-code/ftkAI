
import os
import sys
from unittest.mock import MagicMock

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PRM import PRMStore, PRMController, load_or_create, UpdateDelta
from PRM.schema import Priority

def test_refinement():
    print("Testing PRM Refinement Loop Logic...")
    
    # Setup mock store in memory (using a temp file)
    db_path = "/tmp/test_prm.db"
    if os.path.exists(db_path): os.remove(db_path)
    store = PRMStore(db_path)
    state = load_or_create(store, "test_proj", goal="Test refinement")
    
    # Create an echo model that "fails" on the first try
    call_count = 0
    def mock_model(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First response: simulate a bug detection
            return '```json\n{"prm_update": {"new_bugs": [{"summary": "Syntax error in loop", "severity": "high"}]}}\n```'
        else:
            # Second response: resolve the bug
            return '```json\n{"prm_update": {"resolved_bug_ids": ["all"], "goal_confidence": 0.95}}\n```'

    ctrl = PRMController(model_fn=mock_model, store=store, project_id=state.project_id)
    
    print("Executing step_with_refinement...")
    result = ctrl.step_with_refinement("Write a python loop", max_refinement_steps=1)
    
    print(f"Model calls: {call_count}")
    print(f"Final Goal Confidence: {result.updated_state.goal_confidence}")
    
    if call_count == 2 and result.updated_state.goal_confidence > 0.9:
        print("✅ SUCCESS: Refinement loop triggered and corrected the state.")
    else:
        print(f"❌ FAILURE: Expected 2 calls, got {call_count}. Confidence: {result.updated_state.goal_confidence}")
    
    store.close()
    if os.path.exists(db_path): os.remove(db_path)

if __name__ == "__main__":
    test_refinement()
