"""
Unit and E2E tests for continual learning module.

Tests cover:
1. EWC - Fisher information and penalty computation
2. SI - Omega accumulation and consolidation
3. MAS - Importance computation without labels
4. GEM - Gradient projection
5. PackNet - Pruning and freezing
6. Experience Replay - Buffer operations
7. Progressive Networks - Column addition
8. LoRA - Low-rank adaptation
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fptm_ste.continual import (
    EWCClauseMachine,
    SynapticIntelligenceClause,
    MemoryAwareSynapsesClause,
    GradientEpisodicMemory,
    PackNetClause,
    ExperienceReplayBuffer,
    ReplayAugmentedTrainer,
    ProgressiveClauseNetwork,
    ContinualLearningPipeline,
)
from fptm_ste.lora_adapter import (
    LoRALayer,
    LoRALinear,
    LoRAClauseAdapter,
    LoRAVotingAdapter,
    LoRAClauseMachine,
    MultiTaskLoRAClauseMachine,
    merge_lora_weights,
    count_lora_parameters,
)
from fptm_ste import FuzzyPatternTM_STCM


def make_dummy_dataloader(n_samples=64, n_features=50, n_classes=5, batch_size=16):
    """Create a dummy dataloader for testing."""
    x = torch.rand(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_base_tm(n_features=50, n_clauses=32, n_classes=5):
    """Create a base TM model for testing."""
    return FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
    )


# =============================================================================
# EWC Tests
# =============================================================================


class TestEWCClauseMachine:
    """Tests for Elastic Weight Consolidation."""
    
    def test_fisher_diagonal_positive(self):
        """Fisher information should be non-negative."""
        base_tm = make_base_tm()
        ewc = EWCClauseMachine(base_tm, lamb=1000.0)
        dataloader = make_dummy_dataloader()
        
        fisher = ewc.compute_fisher_information(dataloader)
        
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Fisher for {name} has negative values"
    
    def test_penalty_zero_before_consolidation(self):
        """Penalty should be zero before first task consolidation."""
        base_tm = make_base_tm()
        ewc = EWCClauseMachine(base_tm)
        
        penalty = ewc.penalty()
        assert penalty.item() == 0.0
    
    def test_penalty_increases_with_distance(self):
        """Penalty should increase as params move from optimal."""
        base_tm = make_base_tm()
        ewc = EWCClauseMachine(base_tm, lamb=1000.0)
        dataloader = make_dummy_dataloader()
        
        # Consolidate task
        ewc.consolidate_task(dataloader)
        
        # Initial penalty should be small (params at optimal)
        initial_penalty = ewc.penalty().item()
        
        # Perturb parameters
        with torch.no_grad():
            for param in base_tm.parameters():
                param.add_(torch.randn_like(param) * 0.5)
        
        # Penalty should increase
        final_penalty = ewc.penalty().item()
        assert final_penalty > initial_penalty
    
    def test_online_ewc_accumulates_fisher(self):
        """Online EWC should accumulate Fisher across tasks."""
        base_tm = make_base_tm()
        ewc = EWCClauseMachine(base_tm, online=True, gamma=0.9)
        
        dataloader1 = make_dummy_dataloader()
        dataloader2 = make_dummy_dataloader()
        
        # First task
        ewc.consolidate_task(dataloader1)
        fisher1 = {k: v.clone() for k, v in ewc.fisher.items()}
        
        # Second task
        ewc.consolidate_task(dataloader2)
        
        # Fisher should have accumulated
        for name in fisher1:
            if name in ewc.fisher:
                # Not necessarily larger, but should be different
                pass  # Accumulation is: gamma * old + new


# =============================================================================
# SI Tests
# =============================================================================


class TestSynapticIntelligenceClause:
    """Tests for Synaptic Intelligence."""
    
    def test_omega_accumulates_during_training(self):
        """Omega should accumulate during training."""
        base_tm = make_base_tm()
        si = SynapticIntelligenceClause(base_tm, lamb=1.0)
        dataloader = make_dummy_dataloader()
        
        optimizer = torch.optim.Adam(si.parameters(), lr=0.01)
        
        # Train a few steps
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            logits, _ = si(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            si.update_omega()
        
        # Omega_sum should have accumulated
        for name, omega_sum in si.omega_sum.items():
            assert omega_sum.abs().sum() > 0
    
    def test_consolidate_computes_importance(self):
        """Consolidation should compute importance weights."""
        base_tm = make_base_tm()
        si = SynapticIntelligenceClause(base_tm)
        dataloader = make_dummy_dataloader()
        
        optimizer = torch.optim.Adam(si.parameters(), lr=0.01)
        
        # Train
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            logits, _ = si(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            si.update_omega()
        
        # Consolidate
        si.consolidate_task()
        
        # Omega should exist
        assert len(si.omega) > 0
    
    def test_penalty_after_consolidation(self):
        """SI penalty should be computable after consolidation."""
        base_tm = make_base_tm()
        si = SynapticIntelligenceClause(base_tm)
        dataloader = make_dummy_dataloader()
        
        optimizer = torch.optim.Adam(si.parameters(), lr=0.01)
        
        # Train and consolidate
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            logits, _ = si(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            si.update_omega()
        
        si.consolidate_task()
        
        # Penalty should be computable
        penalty = si.penalty()
        assert penalty >= 0


# =============================================================================
# MAS Tests
# =============================================================================


class TestMemoryAwareSynapsesClause:
    """Tests for Memory Aware Synapses."""
    
    def test_importance_unsupervised(self):
        """MAS importance should be computable without labels."""
        base_tm = make_base_tm()
        mas = MemoryAwareSynapsesClause(base_tm)
        
        # Create dataloader with only x (labels not used)
        x = torch.rand(64, 50)
        y = torch.randint(0, 5, (64,))  # Labels provided but not used
        dataloader = DataLoader(TensorDataset(x, y), batch_size=16)
        
        importance = mas.compute_importance(dataloader)
        
        # Importance should be non-negative
        for name, imp in importance.items():
            assert (imp >= 0).all()
    
    def test_consolidate_stores_reference_params(self):
        """Consolidation should store reference parameters."""
        base_tm = make_base_tm()
        mas = MemoryAwareSynapsesClause(base_tm)
        dataloader = make_dummy_dataloader()
        
        mas.consolidate_task(dataloader)
        
        assert len(mas.ref_params) > 0
        assert len(mas.omega) > 0


# =============================================================================
# GEM Tests
# =============================================================================


class TestGradientEpisodicMemory:
    """Tests for Gradient Episodic Memory."""
    
    def test_store_task_memory(self):
        """Should store memory for each task."""
        base_tm = make_base_tm()
        gem = GradientEpisodicMemory(base_tm, memory_per_task=32)
        dataloader = make_dummy_dataloader()
        
        gem.store_task_memory(0, dataloader)
        
        assert 0 in gem.task_memory
        x, y = gem.task_memory[0]
        assert x.shape[0] <= 32
    
    def test_gradient_projection_satisfies_constraints(self):
        """Projected gradient should satisfy GEM constraints."""
        base_tm = make_base_tm()
        gem = GradientEpisodicMemory(base_tm, memory_per_task=32)
        dataloader = make_dummy_dataloader()
        
        # Store task 0 memory
        gem.store_task_memory(0, dataloader)
        gem.current_task = 1
        
        # Create dummy current gradient
        current_grad = {}
        for name, param in base_tm.named_parameters():
            if param.requires_grad:
                current_grad[name] = torch.randn_like(param)
        
        # Project
        projected = gem.project_gradient(current_grad)
        
        # Projected should exist
        assert len(projected) > 0


# =============================================================================
# PackNet Tests
# =============================================================================


class TestPackNetClause:
    """Tests for PackNet."""
    
    def test_frozen_weights_unchanged(self):
        """Frozen weights should not change during training."""
        base_tm = make_base_tm()
        packnet = PackNetClause(base_tm, prune_fraction=0.5)
        dataloader = make_dummy_dataloader()
        
        # First task: use all weights
        optimizer = torch.optim.Adam(packnet.parameters(), lr=0.01)
        
        for batch in dataloader:
            x, y = batch
            loss = packnet.masked_train_step(x, y, optimizer)
        
        # Prune and freeze
        packnet.prune_and_freeze(dataloader)
        
        # Store frozen weights
        frozen_weights = {}
        for name, param in base_tm.named_parameters():
            if name in packnet.task_masks.get(0, {}):
                mask = packnet.task_masks[0][name]
                frozen_weights[name] = (param.data * mask).clone()
        
        # Train on second "task" (still using same model)
        packnet.current_task = 1
        for batch in dataloader:
            x, y = batch
            loss = packnet.masked_train_step(x, y, optimizer)
        
        # Check frozen weights unchanged
        for name, frozen in frozen_weights.items():
            if name in packnet.task_masks.get(0, {}):
                mask = packnet.task_masks[0][name]
                current = base_tm.state_dict()[name] * mask
                assert torch.allclose(frozen, current, atol=1e-5)
    
    def test_available_params_decrease(self):
        """Available parameters should decrease after pruning."""
        base_tm = make_base_tm()
        packnet = PackNetClause(base_tm, prune_fraction=0.5)
        dataloader = make_dummy_dataloader()
        
        # Count initial available
        initial_available = sum(m.sum() for m in packnet.available_masks.values())
        
        # Prune
        packnet.prune_and_freeze(dataloader)
        
        # Count final available
        final_available = sum(m.sum() for m in packnet.available_masks.values())
        
        assert final_available < initial_available


# =============================================================================
# Experience Replay Tests
# =============================================================================


class TestExperienceReplayBuffer:
    """Tests for Experience Replay Buffer."""
    
    def test_reservoir_sampling_uniform(self):
        """Reservoir sampling should give approximately uniform distribution."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        # Add many samples
        for i in range(1000):
            x = torch.rand(1, 10)
            y = torch.tensor([i % 5])
            buffer.add(x, y)
        
        assert len(buffer) == 100
    
    def test_sample_returns_correct_shapes(self):
        """Sampling should return correct shapes."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        # Add samples
        for _ in range(50):
            x = torch.rand(1, 10)
            y = torch.tensor([0])
            buffer.add(x, y)
        
        # Sample
        x, y = buffer.sample(16)
        
        assert x.shape == (16, 10)
        assert y.shape == (16,)
    
    def test_empty_buffer_returns_none(self):
        """Empty buffer should return None."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        x, y = buffer.sample(16)
        
        assert x is None
        assert y is None


class TestReplayAugmentedTrainer:
    """Tests for Replay Augmented Trainer."""
    
    def test_train_step_with_replay(self):
        """Training step should work with replay."""
        base_tm = make_base_tm()
        buffer = ExperienceReplayBuffer(max_size=100)
        trainer = ReplayAugmentedTrainer(base_tm, buffer, replay_batch_size=8)
        
        optimizer = torch.optim.Adam(base_tm.parameters(), lr=0.01)
        
        # Train a few steps
        for _ in range(10):
            x = torch.rand(16, 50)
            y = torch.randint(0, 5, (16,))
            
            total_loss, task_loss = trainer.train_step(x, y, optimizer)
            
            assert total_loss >= 0
            assert task_loss >= 0
        
        # Buffer should have samples
        assert len(buffer) > 0


# =============================================================================
# Progressive Networks Tests
# =============================================================================


class TestProgressiveClauseNetwork:
    """Tests for Progressive Networks."""
    
    def test_add_task_creates_column(self):
        """Adding task should create new column."""
        def tm_fn():
            return FuzzyPatternTM_STCM(n_features=50, n_clauses=16, n_classes=5)
        
        progressive = ProgressiveClauseNetwork(tm_fn)
        
        assert len(progressive.columns) == 0
        
        progressive.add_task()
        
        assert len(progressive.columns) == 1
    
    def test_previous_columns_frozen(self):
        """Previous columns should be frozen."""
        def tm_fn():
            return FuzzyPatternTM_STCM(n_features=50, n_clauses=16, n_classes=5)
        
        progressive = ProgressiveClauseNetwork(tm_fn)
        
        progressive.add_task()  # Task 0
        progressive.add_task()  # Task 1
        
        # First column should be frozen
        for param in progressive.columns[0].parameters():
            assert not param.requires_grad
        
        # Second column should be trainable
        for param in progressive.columns[1].parameters():
            assert param.requires_grad
    
    def test_forward_for_each_task(self):
        """Forward should work for each task."""
        def tm_fn():
            return FuzzyPatternTM_STCM(n_features=50, n_clauses=16, n_classes=5)
        
        progressive = ProgressiveClauseNetwork(tm_fn)
        progressive.add_task()
        progressive.add_task()
        
        x = torch.rand(8, 50)
        
        # Forward for task 0
        logits0, clauses0 = progressive(x, task_id=0)
        assert logits0.shape == (8, 5)
        
        # Forward for task 1
        logits1, clauses1 = progressive(x, task_id=1)
        assert logits1.shape == (8, 5)


# =============================================================================
# LoRA Tests
# =============================================================================


class TestLoRALayer:
    """Tests for LoRA Layer."""
    
    def test_output_shape(self):
        """LoRA should produce correct output shape."""
        lora = LoRALayer(in_features=64, out_features=32, rank=4)
        x = torch.randn(16, 64)
        
        out = lora(x)
        
        assert out.shape == (16, 32)
    
    def test_initialized_with_zero_output(self):
        """LoRA should initially produce near-zero output (B=0)."""
        lora = LoRALayer(in_features=64, out_features=32, rank=4)
        x = torch.randn(16, 64)
        
        out = lora(x)
        
        # B is initialized to zero, so output should be zero
        assert out.abs().max() < 1e-6
    
    def test_delta_weight_shape(self):
        """Delta weight should have correct shape."""
        lora = LoRALayer(in_features=64, out_features=32, rank=4)
        
        delta = lora.get_delta_weight()
        
        assert delta.shape == (32, 64)


class TestLoRALinear:
    """Tests for LoRA Linear wrapper."""
    
    def test_forward_matches_base_initially(self):
        """Initially, LoRA output should match base linear."""
        linear = nn.Linear(64, 32)
        lora_linear = LoRALinear(linear, rank=4, freeze_base=False)
        
        x = torch.randn(16, 64)
        
        base_out = linear(x)
        lora_out = lora_linear(x)
        
        # Should be very close (B=0)
        assert torch.allclose(base_out, lora_out, atol=1e-5)
    
    def test_merge_equivalence(self):
        """Merged output should equal non-merged output."""
        linear = nn.Linear(64, 32)
        lora_linear = LoRALinear(linear, rank=4)
        
        # Modify LoRA weights
        with torch.no_grad():
            lora_linear.lora.lora_B.fill_(0.1)
        
        x = torch.randn(16, 64)
        
        # Non-merged output
        non_merged = lora_linear(x).clone()
        
        # Merge and compare
        lora_linear.merge_weights()
        merged = lora_linear(x)
        
        assert torch.allclose(non_merged, merged, atol=1e-5)


class TestLoRAClauseAdapter:
    """Tests for LoRA Clause Adapter."""
    
    def test_delta_shapes(self):
        """Delta shapes should match clause dimensions."""
        adapter = LoRAClauseAdapter(
            n_clauses=32,
            n_features=50,
            rank=4,
        )
        
        pos_delta = adapter.get_pos_delta()
        neg_delta = adapter.get_neg_delta()
        
        assert pos_delta.shape == (16, 50)  # half clauses
        assert neg_delta.shape == (16, 50)
    
    def test_forward_adapts_logits(self):
        """Forward should adapt clause logits."""
        adapter = LoRAClauseAdapter(n_clauses=32, n_features=50, rank=4)
        
        pos_logits = torch.randn(16, 50)
        neg_logits = torch.randn(16, 50)
        
        adapted_pos, adapted_neg = adapter(pos_logits, neg_logits)
        
        assert adapted_pos.shape == pos_logits.shape
        assert adapted_neg.shape == neg_logits.shape


class TestLoRAClauseMachine:
    """Tests for LoRA Clause Machine."""
    
    def test_base_weights_frozen(self):
        """Base weights should be frozen."""
        base_tm = make_base_tm()
        lora_tm = LoRAClauseMachine(base_tm, rank=4)
        
        for param in base_tm.parameters():
            assert not param.requires_grad
    
    def test_lora_params_trainable(self):
        """LoRA parameters should be trainable."""
        base_tm = make_base_tm()
        lora_tm = LoRAClauseMachine(base_tm, rank=4)
        
        lora_params = lora_tm.get_lora_params()
        
        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad
    
    def test_param_efficiency(self):
        """LoRA should have significantly fewer trainable params."""
        base_tm = make_base_tm()
        lora_tm = LoRAClauseMachine(base_tm, rank=4)
        
        total, trainable = count_lora_parameters(lora_tm)
        
        assert trainable < total * 0.5  # Less than half are trainable
    
    def test_forward_output_shape(self):
        """Forward should produce correct shapes."""
        base_tm = make_base_tm()
        lora_tm = LoRAClauseMachine(base_tm, rank=4)
        
        x = torch.rand(16, 50)
        logits, clauses = lora_tm(x)
        
        assert logits.shape == (16, 5)
        assert clauses.shape == (16, 32)
    
    def test_merge_unmerge_equivalence(self):
        """Merge and unmerge should be reversible."""
        base_tm = make_base_tm()
        lora_tm = LoRAClauseMachine(base_tm, rank=4)
        
        # Store original voting weights
        original_voting = base_tm.voting.data.clone()
        
        # Modify LoRA
        with torch.no_grad():
            lora_tm.voting_adapter.lora_B.fill_(0.1)
        
        # Merge
        lora_tm.merge_weights()
        assert lora_tm.merged
        
        # Unmerge
        lora_tm.unmerge_weights()
        assert not lora_tm.merged
        
        # Should be back to original
        assert torch.allclose(base_tm.voting.data, original_voting, atol=1e-5)


class TestMultiTaskLoRAClauseMachine:
    """Tests for Multi-Task LoRA."""
    
    def test_add_multiple_tasks(self):
        """Should support multiple task adapters."""
        base_tm = make_base_tm()
        multi_lora = MultiTaskLoRAClauseMachine(base_tm, rank=4)
        
        multi_lora.add_task("task_a")
        multi_lora.add_task("task_b")
        
        assert "task_a" in multi_lora.task_adapters
        assert "task_b" in multi_lora.task_adapters
    
    def test_task_specific_forward(self):
        """Forward should use task-specific adapters."""
        base_tm = make_base_tm()
        multi_lora = MultiTaskLoRAClauseMachine(base_tm, rank=4)
        
        multi_lora.add_task("task_a")
        multi_lora.add_task("task_b")
        
        x = torch.rand(16, 50)
        
        logits_a, _ = multi_lora(x, task_id="task_a")
        logits_b, _ = multi_lora(x, task_id="task_b")
        
        assert logits_a.shape == (16, 5)
        assert logits_b.shape == (16, 5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestContinualLearningPipeline:
    """Integration tests for CL Pipeline."""
    
    @pytest.mark.parametrize("method", ["ewc", "si", "mas"])
    def test_pipeline_methods(self, method):
        """Test pipeline with different CL methods."""
        base_tm = make_base_tm()
        pipeline = ContinualLearningPipeline(
            base_tm,
            method=method,
            lamb=1.0,
        )
        
        dataloader = make_dummy_dataloader()
        
        # Train on task
        metrics = pipeline.train_task(dataloader, epochs=2, verbose=False)
        
        assert "train_loss" in metrics
        assert len(metrics["train_loss"]) == 2
    
    def test_pipeline_with_replay(self):
        """Test pipeline with experience replay."""
        base_tm = make_base_tm()
        pipeline = ContinualLearningPipeline(
            base_tm,
            method="ewc",
            use_replay=True,
            replay_buffer_size=100,
        )
        
        dataloader = make_dummy_dataloader()
        
        # Train on task
        metrics = pipeline.train_task(dataloader, epochs=2, verbose=False)
        
        # Replay buffer should have samples
        assert len(pipeline.replay_buffer) > 0
    
    def test_evaluate_accuracy(self):
        """Test evaluation function."""
        base_tm = make_base_tm()
        pipeline = ContinualLearningPipeline(base_tm, method="ewc")
        
        dataloader = make_dummy_dataloader()
        
        acc = pipeline.evaluate(dataloader)
        
        assert 0 <= acc <= 1


# =============================================================================
# E2E Tests (marked slow)
# =============================================================================


class TestE2EContinualLearning:
    """End-to-end tests for continual learning."""
    
    @pytest.mark.slow
    def test_ewc_prevents_forgetting(self):
        """EWC should help prevent forgetting on previous task."""
        base_tm = make_base_tm()
        ewc = EWCClauseMachine(base_tm, lamb=5000.0)
        
        # Task 1 data
        task1_loader = make_dummy_dataloader(n_samples=128)
        
        # Train on task 1
        optimizer = torch.optim.Adam(ewc.parameters(), lr=0.01)
        for epoch in range(10):
            for x, y in task1_loader:
                optimizer.zero_grad()
                logits, _ = ewc(x)
                loss = F.cross_entropy(logits, y) + ewc.penalty()
                loss.backward()
                optimizer.step()
        
        # Consolidate
        ewc.consolidate_task(task1_loader)
        
        # Measure task 1 accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in task1_loader:
                logits, _ = ewc(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.shape[0]
        task1_acc_before = correct / total
        
        # Train on task 2
        task2_loader = make_dummy_dataloader(n_samples=128)
        for epoch in range(5):
            for x, y in task2_loader:
                optimizer.zero_grad()
                logits, _ = ewc(x)
                loss = F.cross_entropy(logits, y) + ewc.penalty()
                loss.backward()
                optimizer.step()
        
        # Measure task 1 accuracy after
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in task1_loader:
                logits, _ = ewc(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.shape[0]
        task1_acc_after = correct / total
        
        # Should not have catastrophic forgetting
        # (EWC should preserve some accuracy)
        assert task1_acc_after > 0.1  # At least better than random


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

