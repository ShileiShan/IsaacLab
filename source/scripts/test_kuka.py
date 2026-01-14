import numpy as np
import asyncio
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.world import World
from omni.isaac.core.utils.types import ArticulationAction

class KukaRandomTester:
    def __init__(self, robot_prim_path="/World/kuka"):
        self.world = World.instance()
        if self.world is None:
            self.world = World()
        
        self.prim_path = robot_prim_path
        
        if not prim_utils.is_prim_path_valid(robot_prim_path):
            print(f"Error: 路径 {robot_prim_path} 不存在！")
            self.valid = False
            return
        
        # 初始化 Articulation
        self.articulation = Articulation(robot_prim_path)
        self.valid = True

    def _setup_dof_info(self):
        """修复后的识别关节限位逻辑"""
        # 确保初始化
        self.articulation.initialize()
        
        self.dof_names = self.articulation.dof_names
        self.num_dofs = len(self.dof_names)
        
        # 修复点：使用底层 view 的属性获取限位
        # 在新版本中，Articulation 内部维护一个 root_view
        if hasattr(self.articulation, "get_dof_limits"):
            dof_limits = self.articulation.get_dof_limits()
        else:
            # 备选方案：从 view 直接读取
            dof_limits = self.articulation._articulation_view.get_dof_limits()
        
        # dof_limits 形状通常是 (num_dofs, 2)
        self.lower_limits = dof_limits[0, :, 0] if len(dof_limits.shape) > 2 else dof_limits[:, 0]
        self.upper_limits = dof_limits[0, :, 1] if len(dof_limits.shape) > 2 else dof_limits[:, 1]
        
        print(f"成功识别机器人: {self.num_dofs} 个关节")
        print(f"关节名称: {self.dof_names}")
        print(f"限位范围: {list(zip(self.lower_limits, self.upper_limits))}")
        

    def move_random(self, start_index=None, end_index=None):
        """
        随机移动部分关节
        :param start_index: 起始关节索引 (默认0)
        :param end_index: 结束关节索引 (默认None, 即最后一个)
        """
        # 检查是否初始化
        if not self.articulation.handles_initialized:
            self._setup_dof_info()

        # 处理索引默认值
        if start_index is None: start_index = 0
        if end_index is None: end_index = self.num_dofs
        
        # 确保范围有效
        start_index = max(0, start_index)
        end_index = min(self.num_dofs, end_index)
        
        if start_index >= end_index:
            print(f"无效的关节索引范围: [{start_index}, {end_index})")
            return None

        # 获取要操作的关节名称
        active_joint_names = self.dof_names[start_index:end_index]
        print(f"正在驱动关节 [{start_index}-{end_index}]: {active_joint_names}")

        # 生成随机位置 (只针对选定的关节)
        # 注意: apply_action 可以接受 sparse command (joint_indices + joint_positions)
        random_sub_pos = np.random.uniform(
            low=self.lower_limits[start_index:end_index], 
            high=self.upper_limits[start_index:end_index], 
            size=(end_index - start_index,)
        )
        
        # 构造稀疏索引
        indices = np.arange(start_index, end_index, dtype=np.int32)
        
        # 下发动作
        controller = self.articulation.get_articulation_controller()
        if controller:
            action = ArticulationAction(
                joint_positions=random_sub_pos.astype(np.float32),
                joint_indices=indices
            )
            controller.apply_action(action)
            
        return random_sub_pos

async def run_random_test_loop(controller, interval=4.0):
    world = World.instance()
    print(">>> 异步随机测试启动...")
    try:
        # 确保 controller 已经获取到了机器人信息
        if not hasattr(controller, 'num_dofs'):
            controller._setup_dof_info()

        while world and world.is_playing():
            # 示例: 只驱动前 3 个关节
            # target_pos = controller.move_random(start_index=0, end_index=3)
            
            # 示例: 只驱动后 3 个关节
            target_pos = controller.move_random(start_index=controller.num_dofs-10, end_index=None)
            
            # 默认：驱动所有关节
            # target_pos = controller.move_random()
            
            if target_pos is not None:
                print(f"下发目标: {np.round(target_pos, 2)}")
            await asyncio.sleep(interval)
    except Exception as e:
        print(f"循环中断或出错: {e}")
        import traceback
        traceback.print_exc()

def start_test():
    # 路径需根据 Stage 修改
    tester = KukaRandomTester("/World/kuka")
    
    if World.instance().is_playing():
        asyncio.ensure_future(run_random_test_loop(tester))
        print("任务已提交。")
    else:
        print("请先点击 Play！")

# 执行
start_test()