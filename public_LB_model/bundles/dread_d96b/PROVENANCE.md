# Two more of our own full-data CoAtNet-2 checkpoints on the DENSE96 corpus

Both trained BY US on the 4,349 report-labelled competition studies (the 58 expert-labelled
studies excluded from training, used only for checkpoint selection), public CC0 soft labels,
trainer sha256 98be0791abdabec19610b3e9a1817a8d0f58ce008d23532fdde678bb8dfbb30b, --folds 0,
16 epochs, 12 windows/study, batch 8, DropPath 0.1, pairwise rank loss (lambda 0.5, tau 1.0)
masked by the public CC0 label file, top-3 gold58-selected SWA.

Corpus for both: 96 slices at NATIVE 384 (SLOTS 27/21/18/12/18, IMG 384, CROP_MM 140.0
unchanged), verified (4407, 96, 384, 384), 0 errors, 0 empty studies. The companion dataset
kaggle-bundle-dread-d96 holds seed 42 of the same configuration.

  raptor_ft_rsnaD96F43_swa.pt   seed 43, our optimizer constants (bb_lr 3e-5, head_lr 1e-3,
                                weight decay 0.02). The second seed of the DENSE96 pair.
  raptor_ft_rsnaHPF42_swa.pt    seed 42, the OPTIMIZER CONSTANTS PUBLISHED by another team
                                (bb_lr 4.01e-5, head_lr 0.00195, weight decay 0.0588). Same
                                corpus and everything else as rsnaD96F42, so the pair isolates
                                those three constants. Their batch-1-with-accumulation-16
                                schedule was NOT copied -- our trainer has no accumulation path,
                                so the effective batch differs (16 studies for them, 8 for us).

No external data, no third-party API, no image left this machine.
Reported, NOT gates: gold58 SWA 0.9047 (D96F43) and 0.9123 (HPF42) against seed 42's 0.9125.
gold58 is a 58-study development set that has mispredicted this competition's leaderboard four
times, most recently getting the SIGN wrong on DENSE96 (gold58 -0.0019, LB +0.006).
Registered: 2026-10-08_DENSE96_native384.md and 2026-10-08_HP_published.md
