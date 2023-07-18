from training import *

def set_training():
    start_logging()
    device, params = prepare_system()
    if params['CLEAN'] == True:
        dataset, cond_names = prepare_pandas_datasets(params)
        data = prepare_pyg_datasets(params, dataset, cond_names)
        model = perform_training_loop(device, params, data, cond_names, dataset)
    else:
        dataset, clean_dataset, cond_names = prepare_pandas_datasets(params)
        data = prepare_pyg_datasets(params, dataset, cond_names, clean_dataset)
        model = perform_training_loop(device, params, data, cond_names, dataset, clean_dataset)
    torch.save(model.state_dict(), f'../models/model-{params["TARGET_FEATURE_NAME"]}.pt')
    return model

set_training()