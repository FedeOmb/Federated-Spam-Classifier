configs_fedavg5 = [
    {'local_epochs': 5, 'lr': 0.5, 'lambda_reg': 0.001},
    {'local_epochs': 5, 'lr': 1.0, 'lambda_reg': 0.001},
    {'local_epochs': 5, 'lr': 2.0, 'lambda_reg': 0.001},
    {'local_epochs': 5, 'lr': 3.0, 'lambda_reg': 0.001},
    {'local_epochs': 10, 'lr': 0.5, 'lambda_reg': 0.001},
    {'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.001},
    {'local_epochs': 10, 'lr': 2.0, 'lambda_reg': 0.001},
    {'local_epochs': 10, 'lr': 3.0, 'lambda_reg': 0.001},
    {'local_epochs': 20, 'lr': 0.5, 'lambda_reg': 0.001},
    {'local_epochs': 20, 'lr': 1.0, 'lambda_reg': 0.001},
    {'local_epochs': 20, 'lr': 2.0, 'lambda_reg': 0.001},
    {'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001},
]

configs_fedprox3= [
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 1.0},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 1.0},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 1.0},
    ]

configs_fedprox4= [
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.001},
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.01},
    {'local_epochs': 20, 'lr': 0.1, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 0.5, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 1.0, 'mu': 0.1},
    {'local_epochs': 20, 'lr': 2.0, 'mu': 0.1},
    ]

configs_fedadmm1 = [
    {'local_epochs': 20, 'lr': 0.1, 'rho': 0.01},
    {'local_epochs': 20, 'lr': 0.5, 'rho': 0.01},
    {'local_epochs': 20, 'lr': 1.0, 'rho': 0.01},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 0.01},
    {'local_epochs': 20, 'lr': 0.1, 'rho': 0.1},
    {'local_epochs': 20, 'lr': 0.5, 'rho': 0.1},
    {'local_epochs': 20, 'lr': 1.0, 'rho': 0.1},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 0.1},
    {'local_epochs': 20, 'lr': 0.1, 'rho': 1.0},
    {'local_epochs': 20, 'lr': 0.5, 'rho': 1.0},
    {'local_epochs': 20, 'lr': 1.0, 'rho': 1.0},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 1.0},
    {'local_epochs': 20, 'lr': 0.1, 'rho': 2.0},
    {'local_epochs': 20, 'lr': 0.5, 'rho': 2.0},
    {'local_epochs': 20, 'lr': 1.0, 'rho': 2.0},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 2.0},
    ]

configs_fedadmm2 = [
    {'local_epochs': 20, 'lr': 2.0, 'rho': 0.01, 'lambda_l1': 0.001},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 0.01, 'lambda_l1': 0.01},
    {'local_epochs': 20, 'lr': 2.0, 'rho': 0.01, 'lambda_l1': 0},
    ]

best_params_valid = {
     'fedavg':   {'rounds': 200, 'local_epochs': 20, 'batch_size': 64, 'lr': 3.0, 'lambda_reg': 0.001},
     'fedprox':  {'rounds': 200, 'local_epochs': 20, 'batch_size': 64, 'lr': 2.0, 'mu': 0.001,  'lambda_reg': 0.001},
     'fedadmm':  {'rounds': 200, 'local_epochs': 20, 'batch_size': 64, 'lr': 2.0, 'rho': 0.01, 'lambda_l1': 0.001,  'lambda_l2': 0.001}
 }

