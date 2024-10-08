from monai.apps.auto3dseg import AutoRunner
work_dir = "./work_dir"
input = "./task.yaml"
algos = ["segresnet"]

runner = AutoRunner( work_dir=work_dir, input=input, algos=algos)
runner.set_num_fold(1) # set fold in data.json =0 to train with 1 fold
runner.set_device_info(cuda_visible_devices=[0])
# runner.ensemble = False
runner.train = True
runner.algo_gen = False
runner.analyze = False

runner.run()