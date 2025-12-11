configs_fedavg1 = [
    {'rounds': 100, 'local_epochs': 10, 'lr': 0.5, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 10, 'lr': 2.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 10, 'lr': 3.0, 'lambda_reg': 0.001}, #best
]

configs_fedavg2 = [
    {'rounds': 100, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.01},
    {'rounds': 100, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.0001}, #best
    {'rounds': 100, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.1},
]
configs_fedavg3 = [
    {'rounds': 100, 'local_epochs': 5, 'lr': 3.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 10, 'lr': 3.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001},
    {'rounds': 100, 'local_epochs': 30, 'lr': 3.0, 'lambda_reg': 0.001}, #best
]

configs_fedavg4 = [
    {'rounds': 100, 'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001, 'batch_size': 32},
    {'rounds': 100, 'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001, 'batch_size': 64},
    {'rounds': 100, 'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001, 'batch_size': 128},
    {'rounds': 100, 'local_epochs': 20, 'lr': 3.0, 'lambda_reg': 0.001, 'batch_size': 16}, #best
]

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


configs_fedprox1 = [
    {'rounds': 200, 'local_epochs': 10, 'lr': 0.1, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 0.5, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 1.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 2.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.01},
    ]
configs_fedprox2 = [
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.001},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.1},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 1.0},
    ]