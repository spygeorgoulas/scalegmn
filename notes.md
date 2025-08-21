-

# exp scalegmn perm
## scalegm
python inr_classification.py --conf configs/_3d_inr_cls/scalegmn.yml --wandb True

## nfn_1x128_1k_rwi_pth
WANDB_AGENT_MAX_INITIAL_FAILURES=1000 wandb agent mscai-spygeorgoulas/_3d_inr_cls/z7epuzdx

# exp scalegmn scale
## nfn_1x128_1k_rwi_pth
WANDB_AGENT_MAX_INITIAL_FAILURES=1000 wandb agent mscai-spygeorgoulas/_3d_inr_cls/gpu83cj9
WANDB_AGENT_MAX_INITIAL_FAILURES=1000 wandb agent mscai-spygeorgoulas/_3d_inr_cls/ip9a6c4c