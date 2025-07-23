import concurrent
from guangxi_proj_infer import main as infer_main
from inference_proc.fileman import *
from inference_proc.graph_optimize import GraphModel
from inference_proc.mesh_gener import gen_surface
from pathlib import Path
import dataman.data_preprocessor.main as preproc
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils.custom_errors import DataUnavailableError
import time
from utils.analtime import analyze_timestamps

####################可以按需修改的路径########################
# 推理时所用到的权重文件
INPUT_PTHFILE = Path('pth_files/checkpoint_guangxi.pth')
# 输入数据的目录，一般不修改
INPUT_ROOF_XYZ_DIR = Path('input/roof_xyz_files')
INPUT_FULL_XYZ_DIR = Path('input/full_xyz_files')
# 输出结果的目录
OUTPUT_DIR = Path('output')
OUTPUT_WIREFRAME_DIR = OUTPUT_DIR / 'roof_wireframe'
OUTPUT_COMP_MESH_DIR = OUTPUT_DIR / 'complete_mesh'
OUTPUT_ROOF_MESH_DIR = OUTPUT_DIR / 'roof_mesh'
###########################################################

DATA_BASE = Path('data')
EXPR_BASE = Path('experiments')
INFER_DATASET_NAME = 'infer-temp'
INFER_DATASET_DIR = DATA_BASE / INFER_DATASET_NAME
INFER_XYZ_DIR = INFER_DATASET_DIR / 'raw' / 'test' / 'xyz'
INFER_PROCESSED_DIR = INFER_DATASET_DIR / 'processed'
INFER_RESULT_DIR = EXPR_BASE / 'result' / INFER_DATASET_NAME
TEST_LIST_FILE = INFER_PROCESSED_DIR / 'test_list.txt'
MUTI_PROC_FLAG = False

TASK_COUNTER = {
	'total': 0,
# 每一步的 成功数/总数
	'prepared': (0, 0),
	'preprocessed': (0, 0),
	'inferred': (0, 0),
	'graph_optimized': (0, 0),
	'surface_generated': (0, 0)
}

if __name__ == '__main__':
	input()
	
	ts_start = time.perf_counter()
	# ==============================================================================================
	# 创建推理数据目录结构
	create_data_structure(INFER_DATASET_DIR)
	# 从 input 复制点云文件到 data/infer-temp
	copy_xyz_files(INPUT_ROOF_XYZ_DIR, INFER_XYZ_DIR)
	# 生成 test_list.txt
	record_xyz_filenames(INFER_XYZ_DIR, TEST_LIST_FILE)
	TASK_COUNTER['prepared'] = (len(os.listdir(INFER_XYZ_DIR)), len(os.listdir(INPUT_ROOF_XYZ_DIR)))
	prepare_time = time.perf_counter()
	# ==============================================================================================
	# 从点云文件生成中间文件
	preproc_args = argparse.Namespace()
	preproc_args.dataset_dir = INFER_DATASET_DIR
	preproc_args.tasks = ['rgb']
	preproc_args.img_size = 256
	preproc_args.padding = 24
	preproc.run_unified_processing(preproc_args)
	TASK_COUNTER['preprocessed'] = (len(os.listdir(INFER_XYZ_DIR)), len(os.listdir(INPUT_ROOF_XYZ_DIR)))
	preproc_time = time.perf_counter()
	# ==============================================================================================
	# 开始深度学习推理
	result_dir = infer_main(INFER_DATASET_NAME, INPUT_PTHFILE, 256, 3, False, log_dir="./logs")
	TASK_COUNTER['inferred'] = (len(os.listdir(result_dir)), len(os.listdir(INPUT_ROOF_XYZ_DIR)))
	infer_time = time.perf_counter()
	# ==============================================================================================
	# 调用 graph_optimize
	input_path = Path(result_dir)
	models = [
		GraphModel(input_path / f, OUTPUT_WIREFRAME_DIR / f)
		for f in os.listdir(input_path)
		if f.endswith(".obj")
	]
	for m in tqdm(models, desc="闭合优化"):
		m.execute()
	graphopt_time = time.perf_counter()
	TASK_COUNTER['graph_optimized'] = (len(os.listdir(OUTPUT_WIREFRAME_DIR)), len(os.listdir(result_dir)))
	# ==============================================================================================
	# 调用 GenSurface
	os.makedirs(OUTPUT_COMP_MESH_DIR, exist_ok=True)
	os.makedirs(OUTPUT_ROOF_MESH_DIR, exist_ok=True)
	if MUTI_PROC_FLAG:
		wireframe_files = os.listdir(OUTPUT_WIREFRAME_DIR)
		with ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
			futures = {}
			for f in tqdm(wireframe_files, desc="加载表面生成工作进程"):
				f_name = Path(f).stem
				wireframe_path = OUTPUT_WIREFRAME_DIR / f"{f_name}.obj"
				pointcloud_path = INPUT_FULL_XYZ_DIR / f"{f_name}.xyz"
				output_comp_mesh_path = OUTPUT_COMP_MESH_DIR / f"{f_name}.obj"
				output_roof_mesh_path = OUTPUT_ROOF_MESH_DIR / f"{f_name}.obj"
				if not pointcloud_path.exists():
					continue
				futures[executor.submit(gen_surface, wireframe_path, pointcloud_path, output_comp_mesh_path, output_roof_mesh_path)] = f_name
			with tqdm(futures, total=len(wireframe_files), desc="表面生成") as tbar:
				success_count = 0
				failed_futures = []
				for future in futures:
					f_name = futures[future]
					tbar.set_postfix_str(f"{f_name}")
					tbar.update(1)
					try:
						future.result(timeout=10)
						success_count += 1
					except DataUnavailableError:
						print(f"无法提取表面：{f_name}")
						failed_futures.append(future)
						continue
					except concurrent.futures.TimeoutError:
						# print(f"表面生成超时：{f_name}")
						failed_futures.append(future)
						continue
					except concurrent.futures.process.BrokenProcessPool:
						print(f"{f_name} 进程已退出")
						continue

				print(f"表面生成完成，成功 {success_count} / {len(wireframe_files)}")
				print("失败列表：")
				for future in failed_futures:
					f_name = futures[future]
					print(f"{f_name}")
			print("正在关闭工作进程池...")
			executor.shutdown(wait=False, cancel_futures=True)
		TASK_COUNTER['surface_generated'] = (len(os.listdir(OUTPUT_COMP_MESH_DIR)), len(wireframe_files))
	else:
		with tqdm(os.listdir(OUTPUT_WIREFRAME_DIR), desc="表面生成") as tbar:
			for f in os.listdir(OUTPUT_WIREFRAME_DIR):
				f_name = Path(f).stem
				wireframe_path = OUTPUT_WIREFRAME_DIR / f"{f_name}.obj"
				pointcloud_path = INPUT_FULL_XYZ_DIR / f"{f_name}.xyz"
				output_comp_mesh_path = OUTPUT_COMP_MESH_DIR / f"{f_name}.obj"
				output_roof_mesh_path = OUTPUT_ROOF_MESH_DIR / f"{f_name}.obj"
				try:
					gen_surface(wireframe_path, pointcloud_path, output_comp_mesh_path, output_roof_mesh_path)
				except FileNotFoundError:
					continue
				except DataUnavailableError:
					continue
				tbar.set_postfix_str(f"{f_name}")
				tbar.update(1)
		TASK_COUNTER['surface_generated'] = (len(os.listdir(OUTPUT_COMP_MESH_DIR)), len(os.listdir(OUTPUT_WIREFRAME_DIR)))


	print("--- 重建完成 ---")
	print(f"成功 {len(os.listdir(OUTPUT_COMP_MESH_DIR))} / {len(os.listdir(INPUT_ROOF_XYZ_DIR))}")
	# input("按任意键退出...")
	gensurface_time = time.perf_counter()
	print("--- 删除临时文件 ---")
	# 删除临时文件
	recursive_delete_directory(INFER_RESULT_DIR)
	recursive_delete_directory(INFER_DATASET_DIR)
	delete_time = time.perf_counter()

	analyze_timestamps([
		("Start", ts_start, None),
		("Data Prepare", prepare_time, TASK_COUNTER['prepared'][0]),
		("Data Preproc", preproc_time, TASK_COUNTER['preprocessed'][0]),
		("Deep Inference", infer_time, TASK_COUNTER['inferred'][0]),
		("Graph Optimize", graphopt_time, TASK_COUNTER['graph_optimized'][0]),
		("Surface Generate", gensurface_time, TASK_COUNTER['surface_generated'][0]),
		("Clear", delete_time, None),
	])
