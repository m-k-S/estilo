from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
import json
import os
from argparse import Namespace
import np

from util import util
from util import MLS

# with open(opt.json_args, "r") as f:
#     opt = Namespace(**json.loads(f.read()))

opt_dict = '''{
  "datarootA": "nekobasu.jpg",
  "datarootB": "mtabus.jpg",
  "gpu_ids": [0],
  "name": "bus",
  "k_final": 8,
  "k_per_level": 10,
  "tau": 0.05,
  "border_size": 7,
  "input_nc": 3,
  "batchSize": 1,
  "imageSize": 224,
  "fast": true,
  "results_dir": "./results",
  "save_path": "./",
  "niter_decay": 100,
  "beta1": 0.5,
  "lr": 0.05,
  "gamma": 1,
  "convergence_threshold": 0.001
}'''

opt = Namespace(**json.loads(opt_dict))
vgg19 = vgg19_model.define_Vgg19(opt)

save_dir = os.path.join(opt.results_dir, opt.name)
nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau, opt.border_size, save_dir, opt.k_per_level, opt.k_final, opt.fast)
A = util.read_image(opt.datarootA, opt.imageSize)
B = util.read_image(opt.datarootB, opt.imageSize)
points = nbbs.run(A, B)
mls = MLS.MLS(v_class=np.int32)
mls.run_MLS_in_folder(root_folder=save_dir)
