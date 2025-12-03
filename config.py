from torch import nn

class config:
    SEED = 1234
    train_data = "train"   #train  test_100
    eval_data = "val" #"val"
    test_data = "test"  # "test"
    check_data = "test_100"  # "test"
    device = "cuda"
    file_name_model = "model"
    file_name_plot = "losses"
    batch_size = 512 #512 #256
    exp_name = "exp_1"
    epochs = 3000 #3000
    time_dim = 64

    # Graph
    hidden = 256
    net_layers = 6
    act = nn.SiLU()
    attention = True
    tanh = True
    context_nodes = 0
    norm_constant = 1
    invariant_sublayers = 1

    # Diffusion
    timesteps = 1000
    noise_schedule = 'cosine' #NoiseScheduleTypes.POLYNOMIAL
    noise_precision = 1e-5
    diff_loss_type = "l2"
    
    # Common
    ode_reg = 0.001
    clip_grad = True
    ema_decay = 0.999
    include_charges = True
    embedding = False
    norm_factors = [1, 4, 1]
    norm_factor = 1
    agg_method = "sum"
