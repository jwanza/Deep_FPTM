"""
Temporal Clause Machine.

Extends Tsetlin Machines with temporal processing capabilities,
allowing clauses to maintain hidden states and process sequences.

Key Innovations:
1. State-Based Clauses - Clauses have memory that persists across time steps
2. Temporal Attention - Attend to relevant past clause activations
3. Gated State Updates - LSTM/GRU-style gating for clause states
4. Clause History - Maintain history of clause activations

Use Cases:
- Time series classification
- Video understanding
- Sequential pattern recognition
- Natural language processing

Architecture:
- Input at each time step processed by clauses
- Clause states updated based on current input and previous state
- Temporal attention aggregates information across time
- Final prediction from aggregated clause outputs
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Clause State Encoder
# =============================================================================


class ClauseStateEncoder(nn.Module):
    """
    Encodes clause activations into hidden state space.
    
    Args:
        n_clauses: Number of clauses
        state_dim: Dimension of hidden state
    """
    
    def __init__(
        self,
        n_clauses: int,
        state_dim: int,
    ):
        super().__init__()
        self.encoder = nn.Linear(n_clauses, state_dim)
        self.norm = nn.LayerNorm(state_dim)
    
    def forward(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        """Encode clause outputs to state space."""
        return self.norm(self.encoder(clause_outputs))


# =============================================================================
# Gated State Update
# =============================================================================


class GatedStateUpdate(nn.Module):
    """
    LSTM-style gated update for clause states.
    
    Implements input, forget, and output gates to control
    how new clause activations update the hidden state.
    
    Args:
        state_dim: Dimension of hidden state
        input_dim: Dimension of input (encoded clause outputs)
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        input_dim = input_dim or state_dim
        
        # Input gate
        self.input_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # Forget gate
        self.forget_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # Output gate
        self.output_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # Candidate state
        self.candidate = nn.Linear(state_dim + input_dim, state_dim)
    
    def forward(
        self,
        input: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update hidden state with gating.
        
        Args:
            input: Current encoded clause outputs [batch, input_dim]
            hidden_state: Previous hidden state [batch, state_dim]
            cell_state: Previous cell state [batch, state_dim]
            
        Returns:
            (new_hidden_state, new_cell_state)
        """
        if cell_state is None:
            cell_state = torch.zeros_like(hidden_state)
        
        combined = torch.cat([hidden_state, input], dim=-1)
        
        # Gates
        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        
        # Candidate
        c_tilde = torch.tanh(self.candidate(combined))
        
        # Update cell state
        new_cell = f * cell_state + i * c_tilde
        
        # Update hidden state
        new_hidden = o * torch.tanh(new_cell)
        
        return new_hidden, new_cell


class GRUStateUpdate(nn.Module):
    """
    GRU-style gated update for clause states.
    
    Simpler than LSTM with fewer parameters.
    
    Args:
        state_dim: Dimension of hidden state
        input_dim: Dimension of input
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        input_dim = input_dim or state_dim
        
        # Reset gate
        self.reset_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # Update gate
        self.update_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # Candidate
        self.candidate = nn.Linear(state_dim + input_dim, state_dim)
    
    def forward(
        self,
        input: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update hidden state with GRU gating.
        
        Args:
            input: Current encoded clause outputs [batch, input_dim]
            hidden_state: Previous hidden state [batch, state_dim]
            
        Returns:
            new_hidden_state [batch, state_dim]
        """
        combined = torch.cat([hidden_state, input], dim=-1)
        
        # Gates
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        # Reset hidden state
        combined_reset = torch.cat([r * hidden_state, input], dim=-1)
        
        # Candidate
        h_tilde = torch.tanh(self.candidate(combined_reset))
        
        # Update
        new_hidden = (1 - z) * hidden_state + z * h_tilde
        
        return new_hidden


# =============================================================================
# Temporal Attention
# =============================================================================


class TemporalClauseAttention(nn.Module):
    """
    Attention over temporal sequence of clause activations.
    
    Allows the model to focus on relevant past time steps
    when making predictions.
    
    Args:
        state_dim: Dimension of hidden states
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        state_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(state_dim)
    
    def forward(
        self,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            states: Sequence of hidden states [batch, seq_len, state_dim]
            mask: Attention mask [batch, seq_len]
            
        Returns:
            (attended_states, attention_weights)
        """
        attended, weights = self.attention(
            states, states, states,
            key_padding_mask=mask,
        )
        return self.norm(states + attended), weights


# =============================================================================
# Temporal Clause Machine
# =============================================================================


class TemporalClauseMachine(nn.Module):
    """
    Temporal Clause Machine for sequence processing.
    
    Processes sequences of inputs through a Tsetlin Machine with
    temporal hidden states that maintain context across time steps.
    
    Args:
        n_features: Number of input features per time step
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        state_dim: Dimension of hidden state
        state_update: Type of state update ('lstm', 'gru')
        use_temporal_attention: Apply attention over time steps
        n_attention_heads: Number of attention heads
        pooling: Final pooling strategy ('last', 'mean', 'max', 'attention')
        operator: TM clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        state_dim: int = 64,
        state_update: str = "gru",
        use_temporal_attention: bool = True,
        n_attention_heads: int = 4,
        pooling: str = "last",
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.state_dim = state_dim
        self.pooling = pooling
        
        # TM for processing each time step
        self.tm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
        
        # Clause to state encoder
        self.state_encoder = ClauseStateEncoder(
            n_clauses=n_clauses,
            state_dim=state_dim,
        )
        
        # State update mechanism
        if state_update == "lstm":
            self.state_update = GatedStateUpdate(
                state_dim=state_dim,
                input_dim=state_dim,
            )
            self._use_cell_state = True
        elif state_update == "gru":
            self.state_update = GRUStateUpdate(
                state_dim=state_dim,
                input_dim=state_dim,
            )
            self._use_cell_state = False
        else:
            raise ValueError(f"Unknown state_update: {state_update}")
        
        # Temporal attention
        if use_temporal_attention:
            self.temporal_attention = TemporalClauseAttention(
                state_dim=state_dim,
                n_heads=n_attention_heads,
            )
        else:
            self.temporal_attention = None
        
        # Output projection
        self.output = nn.Linear(state_dim, n_classes)
        
        # Attention pooling (if used)
        if pooling == "attention":
            self.pool_attention = nn.Linear(state_dim, 1)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Initialize hidden state for a batch."""
        hidden = torch.zeros(batch_size, self.state_dim, device=device)
        cell = torch.zeros(batch_size, self.state_dim, device=device) if self._use_cell_state else None
        return hidden, cell
    
    def forward_step(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a single time step.
        
        Args:
            x: Input for this time step [batch, n_features]
            hidden: Previous hidden state [batch, state_dim]
            cell: Previous cell state (for LSTM) [batch, state_dim]
            use_ste: Use straight-through estimator
            
        Returns:
            (clause_outputs, new_hidden, new_cell)
        """
        # TM processing
        _, clauses = self.tm(x, use_ste=use_ste, skip_norm=True)
        
        # Encode clauses
        encoded = self.state_encoder(clauses)
        
        # Update state
        if self._use_cell_state:
            new_hidden, new_cell = self.state_update(encoded, hidden, cell)
        else:
            new_hidden = self.state_update(encoded, hidden)
            new_cell = None
        
        return clauses, new_hidden, new_cell
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        use_ste: bool = True,
        return_all_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Process a sequence.
        
        Args:
            x: Input sequence [batch, seq_len, n_features]
            hidden: Initial hidden state
            cell: Initial cell state (for LSTM)
            use_ste: Use straight-through estimator
            return_all_states: Return all intermediate states
            
        Returns:
            (logits, final_hidden) or dict with all states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden state
        if hidden is None:
            hidden, cell = self.init_hidden(batch_size, device)
        
        # Process sequence
        all_states = []
        all_clauses = []
        
        for t in range(seq_len):
            x_t = prepare_tm_input(x[:, t], n_features=self.n_features)
            clauses, hidden, cell = self.forward_step(x_t, hidden, cell, use_ste)
            all_states.append(hidden)
            all_clauses.append(clauses)
        
        # Stack states: [batch, seq_len, state_dim]
        states = torch.stack(all_states, dim=1)
        
        # Apply temporal attention
        attention_weights = None
        if self.temporal_attention is not None:
            states, attention_weights = self.temporal_attention(states)
        
        # Pool across time
        if self.pooling == "last":
            pooled = states[:, -1]
        elif self.pooling == "mean":
            pooled = states.mean(dim=1)
        elif self.pooling == "max":
            pooled = states.max(dim=1)[0]
        elif self.pooling == "attention":
            attn_scores = self.pool_attention(states).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            pooled = (states * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Output
        logits = self.output(pooled)
        
        if return_all_states:
            return {
                "logits": logits,
                "final_hidden": hidden,
                "final_cell": cell,
                "all_states": states,
                "all_clauses": torch.stack(all_clauses, dim=1),
                "attention_weights": attention_weights,
            }
        
        return logits, hidden


# =============================================================================
# Bidirectional Temporal Clause Machine
# =============================================================================


class BidirectionalTemporalClauseMachine(nn.Module):
    """
    Bidirectional Temporal Clause Machine.
    
    Processes sequences in both forward and backward directions,
    combining information from past and future context.
    
    Args:
        n_features: Number of input features per time step
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        state_dim: Dimension of hidden state (per direction)
        state_update: Type of state update ('lstm', 'gru')
        pooling: Final pooling strategy
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        state_dim: int = 64,
        state_update: str = "gru",
        pooling: str = "last",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pooling = pooling
        
        # Forward TM
        self.forward_tm = TemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_dim=state_dim,
            state_update=state_update,
            use_temporal_attention=False,  # We'll combine directions first
            pooling="last",
        )
        
        # Backward TM
        self.backward_tm = TemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_dim=state_dim,
            state_update=state_update,
            use_temporal_attention=False,
            pooling="last",
        )
        
        # Combined output
        self.output = nn.Linear(state_dim * 2, n_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, n_features]
            use_ste: Use straight-through estimator
            
        Returns:
            (logits, combined_hidden)
        """
        # Forward pass
        _, hidden_fwd = self.forward_tm(x, use_ste=use_ste)
        
        # Backward pass (reverse sequence)
        x_reversed = x.flip(dims=[1])
        _, hidden_bwd = self.backward_tm(x_reversed, use_ste=use_ste)
        
        # Combine
        combined = torch.cat([hidden_fwd, hidden_bwd], dim=-1)
        
        # Output
        logits = self.output(combined)
        
        return logits, combined


# =============================================================================
# Temporal Clause History
# =============================================================================


class ClauseHistoryBuffer:
    """
    Maintains a history of clause activations.
    
    Useful for visualization and analysis of temporal patterns.
    
    Args:
        max_length: Maximum history length
    """
    
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.history: List[torch.Tensor] = []
    
    def add(self, clause_outputs: torch.Tensor) -> None:
        """Add clause outputs to history."""
        self.history.append(clause_outputs.detach().cpu())
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    def get_history(self) -> torch.Tensor:
        """Get stacked history tensor."""
        if not self.history:
            return torch.tensor([])
        return torch.stack(self.history, dim=0)
    
    def clear(self) -> None:
        """Clear history."""
        self.history = []


# =============================================================================
# Streaming Temporal TM
# =============================================================================


class StreamingTemporalClauseMachine(nn.Module):
    """
    Streaming Temporal Clause Machine for online processing.
    
    Maintains state between calls, suitable for real-time
    streaming applications.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        state_dim: Hidden state dimension
        output_every: Output prediction every N steps
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        state_dim: int = 64,
        output_every: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.state_dim = state_dim
        self.output_every = output_every
        
        self.temporal_tm = TemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_dim=state_dim,
            use_temporal_attention=False,
            pooling="last",
        )
        
        # Persistent state
        self._hidden: Optional[torch.Tensor] = None
        self._cell: Optional[torch.Tensor] = None
        self._step_count = 0
    
    def reset_state(self) -> None:
        """Reset internal state."""
        self._hidden = None
        self._cell = None
        self._step_count = 0
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Process a single time step or batch of steps.
        
        Args:
            x: Input [batch, n_features] or [batch, seq_len, n_features]
            use_ste: Use straight-through estimator
            
        Returns:
            Logits if output_every reached, else None
        """
        # Handle single step input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize state if needed
        if self._hidden is None:
            self._hidden, self._cell = self.temporal_tm.init_hidden(batch_size, device)
        
        # Process steps
        for t in range(x.shape[1]):
            x_t = prepare_tm_input(x[:, t], n_features=self.n_features)
            _, self._hidden, self._cell = self.temporal_tm.forward_step(
                x_t, self._hidden, self._cell, use_ste
            )
            self._step_count += 1
        
        # Output if interval reached
        if self._step_count % self.output_every == 0:
            logits = self.temporal_tm.output(self._hidden)
            return logits
        
        return None


# =============================================================================
# Factory Functions
# =============================================================================


def create_temporal_tm(
    n_features: int,
    n_clauses: int,
    n_classes: int,
    sequence_model: str = "gru",
    bidirectional: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create temporal TM variants.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        sequence_model: 'gru' or 'lstm'
        bidirectional: Use bidirectional processing
        **kwargs: Additional arguments
        
    Returns:
        Temporal TM model
    """
    if bidirectional:
        return BidirectionalTemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_update=sequence_model,
            **kwargs,
        )
    else:
        return TemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_update=sequence_model,
            **kwargs,
        )


