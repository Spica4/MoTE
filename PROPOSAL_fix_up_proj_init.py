"""
Proposal: Change up_proj initialization from zeros to small random values

Current issue:
- up_proj is initialized with zeros (LoRA standard)
- With low learning rate, it cannot escape from zero
- down_proj is initialized with Kaiming, so it can be trained

Solution:
Instead of zeros, initialize up_proj with very small random values.

In backbone/swin_unetr_mote.py, line 54:
Change:
    nn.init.zeros_(self.up_proj.weight)
To:
    nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.01)

This gives up_proj a small non-zero starting point while maintaining
the spirit of LoRA (starting with small adapter influence).
"""

# This is just a proposal document, not actual code change
print(__doc__)
