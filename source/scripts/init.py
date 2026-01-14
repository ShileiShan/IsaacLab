from omni.isaac.core import SimulationContext

# 1. 检查是否已经存在 context，如果存在就清理掉
if SimulationContext.instance():
    SimulationContext.instance().clear_instance()

# 2. 重新初始化
sim_context = SimulationContext()
sim_context.initialize()

# 3. 只有在 initialize 之后，Play 按钮相关的底层逻辑才会重新激活
sim_context.play()