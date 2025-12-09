import unittest
import torch
import torch.nn as nn
from fptm_ste.incremental_tm import IncrementalSTCM, IncrementalConfig

class TestVectorizedIncremental(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        # Small configuration for manual verification
        self.config = IncrementalConfig(
            states_num=10,
            include_limit=5,
            T=10.0,
            S=5.0,
            L=100, # Large budget to not interfere
            use_probabilistic_updates=False, # Deterministic updates
            use_sparse_exploration=False,    # No random exploration
        )
        
        self.n_features = 4
        self.n_clauses = 4 # 2 positive, 2 negative (half=2)
        self.n_classes = 2
        
        self.model = IncrementalSTCM(
            n_features=self.n_features,
            n_clauses=self.n_clauses,
            n_classes=self.n_classes,
            config=self.config,
        )
        
        # Manually initialize automaton states to known values
        # All set to include_limit - 1 (4) initially
        initial_val = 4
        self.model.automaton.pos_states.fill_(initial_val)
        self.model.automaton.neg_states.fill_(initial_val)
        self.model.automaton.pos_inv_states.fill_(initial_val)
        self.model.automaton.neg_inv_states.fill_(initial_val)
        
        # Sync parameters to reflect these states
        self.model._sync_automaton_to_params()

    def test_type1_feedback_vectorized(self):
        """Test Type I feedback (Correct Class) logic."""
        
        # Input: x = [1, 0, 1, 0]
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        y = torch.tensor([0]) # Target class 0 (assumed positive logic for simplicity of test)
        
        # Mock clause outputs to force specific feedback paths
        # We need clause_outputs to be > 0 for Type I feedback to trigger on matching clauses
        # Shape [1, 4] (2 pos, 2 neg)
        # Let's say Clause 0 matches, Clause 1 does not match
        clause_outputs = torch.tensor([[1.0, 0.0, 0.0, 0.0]]) 
        
        # Mock logits
        logits = torch.tensor([[10.0, -10.0]])
        
        # Run feedback
        stats = self.model.incremental_feedback(x, y, clause_outputs, logits)
        
        # Check updates for Clause 0 (Matching, Type I)
        # x=[1, 0, 1, 0]
        
        # Reinforce (add 1):
        # pos: where x=1 -> indices 0, 2 should inc
        # pos_inv: where x=0 -> indices 1, 3 should inc
        
        # Suppress (sub 1):
        # pos: where x=0 -> indices 1, 3 should dec (if < limit)
        # pos_inv: where x=1 -> indices 0, 2 should dec (if < limit)
        
        # Initial state was 4. Limit is 5.
        # Indices 0, 2: pos -> 4+1=5 (inc), pos_inv -> 4-1=3 (suppress)
        # Indices 1, 3: pos -> 4-1=3 (suppress), pos_inv -> 4+1=5 (inc)
        
        pos_states = self.model.automaton.pos_states
        pos_inv_states = self.model.automaton.pos_inv_states
        
        # Clause 0
        self.assertEqual(pos_states[0, 0].item(), 5) # Reinforced
        self.assertEqual(pos_states[0, 2].item(), 5) # Reinforced
        self.assertEqual(pos_states[0, 1].item(), 3) # Suppressed
        self.assertEqual(pos_states[0, 3].item(), 3) # Suppressed
        
        self.assertEqual(pos_inv_states[0, 0].item(), 3) # Suppressed
        self.assertEqual(pos_inv_states[0, 2].item(), 3) # Suppressed
        self.assertEqual(pos_inv_states[0, 1].item(), 5) # Reinforced
        self.assertEqual(pos_inv_states[0, 3].item(), 5) # Reinforced
        
        # Clause 1 (Non-matching) should be untouched (exploration is off)
        self.assertTrue((pos_states[1] == 4).all())
        
        print("\nType I Feedback Vectorization Verification Passed!")

    def test_type2_feedback_vectorized(self):
        """Test Type II feedback (Incorrect Class) logic."""
        
        # Input: x = [1, 0, 1, 0]
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        y = torch.tensor([0])
        
        # For Type II, we look at Negative Clauses (indices 2, 3 in 0-based 4-clause system)
        # But wait, the feedback implementation splits clause_outputs into [:half] (pos) and [half:] (neg)
        # Type II applies to clauses in the 'neg' bank that match.
        
        # Let's say Neg Clause 0 (global index 2) matches.
        clause_outputs = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        
        logits = torch.tensor([[-10.0, 10.0]]) # Wrong prediction, strong negative vote
        
        stats = self.model.incremental_feedback(x, y, clause_outputs, logits)
        
        # Check updates for Neg Clause 0
        # Type II: Reinforce to EXCLUDE pattern
        # Reinforce neg where x=0 -> indices 1, 3
        # Reinforce neg_inv where x=1 -> indices 0, 2
        
        neg_states = self.model.automaton.neg_states
        neg_inv_states = self.model.automaton.neg_inv_states
        
        # Neg Clause 0
        # Indices 0, 2 (x=1): neg untouched, neg_inv reinforced
        self.assertEqual(neg_states[0, 0].item(), 4)
        self.assertEqual(neg_inv_states[0, 0].item(), 5)
        
        # Indices 1, 3 (x=0): neg reinforced, neg_inv untouched
        self.assertEqual(neg_states[0, 1].item(), 5)
        self.assertEqual(neg_inv_states[0, 1].item(), 4)
        
        print("\nType II Feedback Vectorization Verification Passed!")

if __name__ == '__main__':
    unittest.main()

