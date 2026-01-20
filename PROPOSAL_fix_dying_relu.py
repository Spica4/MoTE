"""
Proposal: Fix Dying ReLU issue in adapters

Issue identified:
- up_proj weights remain at 0.00000 after 50 epochs
- down_proj is slightly trained but weights are very small (mean ~0.002)
- ReLU may be outputting mostly zeros, blocking gradients to up_proj

Root cause: Dying ReLU
- If down_proj output is negative, ReLU(x) = 0
- Gradient through ReLU is also 0
- up_proj receives no gradient and cannot be trained

Solutions (in order of recommendation):

1. Replace ReLU with Leaky ReLU (RECOMMENDED)
   File: backbone/swin_unetr_mote.py, line 46

   Change:
     self.non_linear_func = nn.ReLU()

   To:
     self.non_linear_func = nn.LeakyReLU(negative_slope=0.01)

   Effect: Allows small gradients even for negative inputs

2. Replace ReLU with GELU (ALTERNATIVE)

   Change:
     self.non_linear_func = nn.ReLU()

   To:
     self.non_linear_func = nn.GELU()

   Effect: Smoother activation, better gradient flow

3. Initialize down_proj bias with positive values (SUPPLEMENTARY)
   File: backbone/swin_unetr_mote.py, line 55

   Change:
     nn.init.zeros_(self.down_proj.bias)

   To:
     nn.init.constant_(self.down_proj.bias, 0.1)

   Effect: Helps ReLU output positive values initially

Recommendation:
Use solution 1 (Leaky ReLU) as it:
- Fixes dying ReLU problem
- Maintains similar behavior to ReLU for positive values
- Is widely used and proven effective
- Minimal computational overhead
"""

print(__doc__)
