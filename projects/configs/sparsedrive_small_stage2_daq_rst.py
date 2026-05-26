_base_ = ['./sparsedrive_small_stage2.py']

work_dir = './work_dirs/sparsedrive_small_stage2_daq_rst'

data = dict(
	train=dict(ann_file='data/infos/nuscenes_infos_train.pkl'),
	val=dict(ann_file='data/infos/daq_data_infos_infe.pkl'),
	test=dict(ann_file='data/infos/daq_data_infos_infe.pkl'),
)

eval_config = dict(
	ann_file='data/infos/daq_data_infos_infe.pkl',
)
