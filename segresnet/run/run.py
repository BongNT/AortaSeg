from monai.apps.auto3dseg import AutoRunner
work_dir = "./work_dir"
input = "./task.yaml"
algos = ["segresnet"]

runner = AutoRunner( work_dir=work_dir, input=input, algos=algos)
runner.set_num_fold(1)
runner.set_device_info(cuda_visible_devices=[2,3])
# runner.ensemble = False
runner.train = True
runner.algo_gen = False

runner.run()