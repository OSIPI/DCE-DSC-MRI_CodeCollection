from mk_utils.convert import run_conversion_script

def on_pre_build(config):
    run_conversion_script()