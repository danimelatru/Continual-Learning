# src/config.py

class Config:
    # CHANGED: Use a standard ViT model instead of CLIP
    MODEL_CHECKPOINT = "google/vit-base-patch16-224" 
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 5e-5
    
    # Paths
    OUTPUT_DIR_FT_A = "./results_ft_A"
    OUTPUT_DIR_FT_B = "./results_ft_B"
    EVAL_RESULTS_DIR = "./eval_results"
    PLOT_FILENAME = "media/comparison_graph.png"