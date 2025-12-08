# https://www.nature.com/articles/s42256-023-00638-0


import torch
import torch.nn as nn
import torch.nn.functional as F

class T4(nn.Module):
    def __init__(self, num_temporal_feats, num_static_feats, num_treatments, num_outputs, 
                 hidden_size=64, embed_size=32, max_seq_len=50):
        """
        T4 Architecture: Encoder-Decoder with Attention for ITE estimation.
        
        Args:
            num_temporal_feats: Number of time-varying covariates (e.g., vitals).
            num_static_feats: Number of static covariates (e.g., age, gender).
            num_treatments: Dimension of treatment (usually 1 for binary).
            num_outputs: Dimension of outcome (e.g., SOFA score).
            hidden_size: LSTM hidden dimension.
            embed_size: Dimension for feature embeddings.
        """
        super(T4, self).__init__()
        
        # --- 1. Embeddings ---
        # Equation (2) & (3): Embed temporal and static features
        self.temp_embed = nn.Linear(num_temporal_feats, embed_size)
        self.static_embed = nn.Linear(num_static_feats, embed_size)
        
        # --- 2. Encoder ---
        # Input: [Embedded_Temporal, Embedded_Static, Prev_Treatment]
        # Note: We add +1 for the treatment dimension (assuming binary/scalar treatment)
        encoder_input_size = embed_size + embed_size + num_treatments
        self.encoder_lstm = nn.LSTM(encoder_input_size, hidden_size, batch_first=True)
        
        # --- 3. Attention Mechanism ---
        # Equation (5): Scores based on h_t, h_s, and h_t * h_s
        self.attn_W = nn.Linear(hidden_size * 3, 1) # Learning the score
        
        # --- 4. Decoder ---
        # Input: [Embedded_Static, Prev_Outcome, Prev_Treatment]
        # In the paper, decoder is initialized with encoder state.
        decoder_input_size = embed_size + num_outputs + num_treatments
        self.decoder_lstm = nn.LSTM(decoder_input_size, hidden_size, batch_first=True)
        
        # --- 5. Prediction Heads ---
        # Equation (9): Outcome Prediction
        # Input: [Decoder_Hidden, Context_Vector, Current_Treatment]
        self.outcome_head = nn.Linear(hidden_size + hidden_size + num_treatments, num_outputs)
        
        # Equation (16): Propensity/Treatment Prediction (for pre-training/balancing)
        self.propensity_head = nn.Linear(hidden_size, num_treatments)

    def encode(self, temporal_covariates, static_covariates, prev_treatments):
        """
        Encodes patient history.
        """
        batch_size, seq_len, _ = temporal_covariates.shape
        
        # Embeddings
        e_x = F.relu(self.temp_embed(temporal_covariates)) # (Batch, Seq, Emb)
        e_d = F.relu(self.static_embed(static_covariates)).unsqueeze(1).repeat(1, seq_len, 1) # (Batch, Seq, Emb)
        
        # Concatenate: [e_x, e_d, a_{t-1}]
        # Assuming prev_treatments is aligned such that index i is treatment at i-1
        rnn_input = torch.cat([e_x, e_d, prev_treatments], dim=-1)
        
        # Encoder Forward
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(rnn_input)
        return encoder_outputs, (h_n, c_n)

    def calculate_attention(self, decoder_hidden, encoder_outputs):
        """
        Calculates context vector using Attention (Eq 5).
        decoder_hidden: Current step hidden state (Batch, 1, Hidden)
        encoder_outputs: All encoder hidden states (Batch, Seq, Hidden)
        """
        # Expand decoder hidden to match encoder sequence length
        seq_len = encoder_outputs.size(1)
        h_t_expanded = decoder_hidden.expand(-1, seq_len, -1)
        
        # Concatenate [h_t, h_s, h_t * h_s]
        # Interaction term (element-wise product) is key in T4 attention
        interaction = h_t_expanded * encoder_outputs
        attn_input = torch.cat([h_t_expanded, encoder_outputs, interaction], dim=-1)
        
        # Score and Weights
        scores = torch.tanh(self.attn_W(attn_input)) # (Batch, Seq, 1)
        attn_weights = F.softmax(scores, dim=1)
        
        # Context Vector (Weighted Sum)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs) # (Batch, 1, Hidden)
        return context

    def forward(self, temporal_x, static_x, history_treatments, 
                future_treatments, future_init_outcome=None):
        """
        Args:
            temporal_x: (Batch, Hist_Len, Feat)
            static_x: (Batch, Feat)
            history_treatments: (Batch, Hist_Len, 1)
            future_treatments: (Batch, Future_Len, 1) - The 'Action' sequence to test
            future_init_outcome: (Batch, 1) - Outcome at last observed step (y_t)
        
        Returns:
            predicted_outcomes: (Batch, Future_Len, 1)
            propensity_scores: (Batch, Hist_Len, 1) - Optional, for balancing
        """
        # 1. Encode
        encoder_outputs, (hidden, cell) = self.encode(temporal_x, static_x, history_treatments)
        
        # 2. Propensity (Optional: predicted based on encoder history)
        # Used for balancing matching pre-training
        propensity_logits = self.propensity_head(encoder_outputs)
        
        # 3. Decode / Forecast
        batch_size = temporal_x.size(0)
        future_len = future_treatments.size(1)
        
        # Initialize decoder input
        # First input is [static, y_t, a_t] (or similar, paper implies autoregressive)
        # If future_init_outcome not provided, use zeros (or last encoder output transformed)
        curr_outcome = future_init_outcome if future_init_outcome is not None else torch.zeros(batch_size, 1).to(temporal_x.device)
        
        predictions = []
        
        for t in range(future_len):
            # Prepare Input: [Static, Prev_Outcome, Prev_Treatment]
            # Note: For T4, we might use the PLANNED treatment for the input to next state logic
            # or the previous step's treatment. Here we use previous predicted outcome.
            
            e_d = F.relu(self.static_embed(static_x)).unsqueeze(1) # (Batch, 1, Emb)
            
            # Get treatment for THIS step (from the plan)
            curr_action = future_treatments[:, t, :].unsqueeze(1)
            
            # Decoder LSTM Input
            dec_input = torch.cat([e_d, curr_outcome.unsqueeze(1), curr_action], dim=-1)
            
            # Step LSTM
            dec_output, (hidden, cell) = self.decoder_lstm(dec_input, (hidden, cell))
            
            # Attention
            context = self.calculate_attention(dec_output, encoder_outputs)
            
            # Predict Outcome (Eq 9)
            # Concatenate [Hidden, Context, Action]
            out_input = torch.cat([dec_output, context, curr_action], dim=-1)
            pred_y = self.outcome_head(out_input) # (Batch, 1, Out)
            
            predictions.append(pred_y)
            
            # Autoregressive update: Use predicted outcome as next input
            curr_outcome = pred_y.squeeze(1)
            
        predicted_outcomes = torch.cat(predictions, dim=1)
        return predicted_outcomes, propensity_logits