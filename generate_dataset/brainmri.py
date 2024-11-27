import os
import json

class BrainDatasetLoader(object):
    CLSNAMES = ['brain']

    def __init__(self, root=''):
        self.root = root
        self.meta_path = f'{root}/data/Brain_AD/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/data/Brain_AD'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}/img')
                    mask_names = os.listdir(f'{cls_dir}/{phase}/{specie}/label')
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_dir}/{phase}/{specie}/img/{img_name}',
                            mask_path=f'{cls_dir}/{phase}/{specie}/label/{mask_names[idx]}',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = BrainDatasetLoader(root='C:/Users/Deept/OneDrive/Desktop/Github/MedVMAD')
    runner.run()
