from mk_utils.convert import run_conversion_script
from mk_utils.copier import copy_assets

def on_pre_build(config):
    run_conversion_script()
    copy_assets()
