from tqdm import tqdm
from utils import *
from heatmaps import CAMComputer


class Tester(object):
    def __init__(self, model, data_loader, config):
        self.config = config
        # get the size of train and evaluation data
        if isinstance(data_loader, dict):
            self.dataset_sizes = {x: len(data_loader[x].dataset) for x in data_loader.keys()}
        self.project_name = config.project_name

        # Device configuration
        # self.device, self.device_ids = prepare_device(config.n_gpu)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.data_loader = data_loader

        # Config
        self.model_type = config.model_type
        self.dataset_name = config.dataset_name
        self.iou_threshold_list = config.iou_threshold_list
        self.mask_path = config.mask_path
        self.metadata_path = config.metadata_path
        self.multi_iou_eval = config.multi_iou_eval

        self.cam_curve_interval = config.cam_curve_interval
        self.multi_contour_eval = config.multi_contour_eval
        self.cam_test_path = config.cam_test_path

        # Model hyper-parameters
        self.epoch = config.epoch
        self.batch_size = config.batch_size


    def test(self):

        # num_example = 0  # number of examples seen
        # batch = 0
        test_corrects = 0

        # ------------------------- Test -------------------------------------
        split = "test"
        result = make_result_dict(split, self.config)
        self.model.eval()
        batch_correct_top1 = []
        batch_correct_top5 = []

        for iter, (data, target, img_ids) in enumerate(tqdm(self.data_loader[split], desc="Test", mininterval=0.1)):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)

                # Predict Class
                _, preds = torch.max(output['logits'], 1)

            correct_top1, correct_top5 = correct_topk(output['logits'], target, topk=(1, 5))
            test_corrects += torch.sum(preds == target.data)
            batch_correct_top1.append(correct_top1)
            batch_correct_top5.append(correct_top5)

        top1_correct = []
        top5_correct = []
        for i, j in zip(batch_correct_top1, batch_correct_top5):
            top1_correct.append(i.tolist())
            top5_correct.append(j.tolist())
        top1_correct = np.array(sum(top1_correct, []), dtype=int)
        top5_correct = np.array(sum(top5_correct, []), dtype=int)

        #
        cam_computer = CAMComputer(
            model=self.model,
            loader=self.data_loader[split],
            metadata_root=os.path.join(self.metadata_path, split),
            mask_root=self.mask_path,
            iou_threshold_list=self.iou_threshold_list,
            dataset_name=self.dataset_name,
            split=split,
            cam_curve_interval=self.cam_curve_interval,
            multi_contour_eval=self.multi_contour_eval,
            cam_path=self.cam_test_path,
        )

        cam_performance, total_iou_list = cam_computer.compute_and_evaluate_cams()
        cam_computer.visualization_cams(self.config.dataset_path['test'], self.config.project_name,
                                        self.config.multi_contour_eval)

        total_gt_known_list = np.where(total_iou_list >= 0.5, 1, 0)
        top5_index = np.sum(total_gt_known_list * top5_correct.reshape(-1, 1), 0).argmax(0)
        optimal_cam_threshold_index = int(0.04/0.01)

        optimal_iou_list = total_gt_known_list[:, optimal_cam_threshold_index]
        top5_optimal = total_gt_known_list[:, top5_index]

        cam_threshold = self.cam_curve_interval * optimal_cam_threshold_index
        gt_loc = cam_performance[1]
        top1_loc = np.sum((optimal_iou_list * top1_correct), 0) / top1_correct.shape[0]
        top5_loc = np.sum((top5_optimal * top5_correct), 0) / top5_correct.shape[0]

        if self.multi_iou_eval or self.dataset_name == 'OpenImages':
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.iou_threshold_list.index(50)]

        test_top1_cls = test_corrects.cpu().numpy() / self.dataset_sizes[split]

        result[split]['loc'] = loc_score
        result[split]['top1_loc'] = top1_loc
        result[split]['top5_loc'] = top5_loc
        result[split]['gt_loc'] = gt_loc
        result[split]['top1_cls'] = test_top1_cls

        # if self.dataset_name in ('CUB', 'ILSVRC'):
        for idx, IOU_THRESHOLD in enumerate(self.iou_threshold_list):
            result[split]['loc_iou'][IOU_THRESHOLD] = cam_performance[idx]

        print("Test:\t[Top1-Loc:{:.4f}, Top5-Loc:{:.4f}, GT-Loc:{:.4f}, Top1-Cls:{:.4f}, CAM_Threshold:{:.4f}, MaxBoxAcc:{:.4f} ([30]:{:.2f}, [50]:{:.2f}, [70]:{:.2f})\n"
              .format(result[split]['top1_loc'], result[split]['top5_loc'], result[split]['gt_loc'], result[split]['top1_cls'],
                      cam_threshold, loc_score, result[split]['loc_iou'][30],
                      result[split]['loc_iou'][50], result[split]['loc_iou'][70]))

        # Save Text
        with open(os.path.join('test_result.txt'), "a") as f:
            print("Test:\t[Top1-Loc:{:.4f}, Top5-Loc:{:.4f}, GT-Loc:{:.4f}, Top1-Cls:{:.4f}, CAM_Threshold:{:.4f}, MaxBoxAcc:{:.4f} ([30]:{:.2f}, [50]:{:.2f}, [70]:{:.2f})\n"
                    .format(result[split]['top1_loc'], result[split]['top5_loc'], result[split]['gt_loc'], result[split]['top1_cls'],
                            cam_threshold, loc_score, result[split]['loc_iou'][30],
                            result[split]['loc_iou'][50], result[split]['loc_iou'][70]), file=f)