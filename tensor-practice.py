#%%
import torch

# %%
x = torch.empty(5,3)
x

# %%
torch.rand(5,3)

# %%
torch.zeros(5,3, dtype=torch.long)

# %%
x = x.new_ones(5,3 ,dtype=torch.float64)
x

# %%
x_like = torch.randn_like(x)
x

# %%
x.size(), x.shape

# %%
y = torch.rand(5,3)
torch.add(x,y)

# %%
y.add_(x)
y

# %%
x.masked_select(x!=0)



# %%
x.view(15)

# %%
