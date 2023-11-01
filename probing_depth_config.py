import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, type=int, help='Step to probe')
    parser.add_argument('--layer_name', required=True, type=str, help='Name of the probed layer')
    parser.add_argument('--block_type', required=True, type=str, help="Name of the probed block")
    parser.add_argument('--output_dir', default="attn1_out", type=str, help="Name of output dir")
    
    parser.add_argument('--postfix', default="self_attn_out", type=str, help="Postfix of the saved files")
    
    parser.add_argument('--probe_type', default="Linear", type=str, help="Type of the probing classifier")

    parser.add_argument('--normalized', default="yes", type=str, help="Whether normalize the depth map or not")
    
    parser.add_argument('--smoothed', default="no", type=str, help="Whether smooth the depth map or not")
    
    parser.add_argument('--lasso', default="no", type=str, help="lasso regularization")
    
    parser.add_argument('--l1_lambda', default="0.1", type=str, help="Lasso lambda")
    
    cfg = parser.parse_args()

    return cfg