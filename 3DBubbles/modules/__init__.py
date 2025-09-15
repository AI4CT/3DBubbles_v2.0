# modules/__init__.py
"""
3DBubbles 核心模块包

该包包含了3DBubbles系统的所有核心功能模块：
- bubble_analysis: 气泡图像分析
- image_generation: 高质量图像生成
- coordinate_transform: 坐标变换
- flow_composition: 流场合成
- bubble_rendering: 气泡渲染
- flow_generator: 流场生成器

所有模块都经过重构，具有清晰的职责分工和良好的模块化设计。
"""

__version__ = "2.0.0"
__author__ = "3DBubbles Team"

# 导入所有核心模块的主要功能，便于外部使用
from .bubble_analysis import (
    analyze_bubble_image,
    analyze_bubble_image_from_array,
    analyze_bubble_image_gpu,
    analyze_bubble_image_from_array_gpu,
    analyze_bubble_batch_gpu,
    analyze_bubble_image_smart,
    analyze_bubble_image_from_array_smart,
    analyze_bubble_batch_smart,
    get_analysis_performance_report,
    reset_analysis_performance,
    calculate_fourier_descriptors,
    calculate_similarity_metrics,
    cosine_similarity_manual,
    TORCH_AVAILABLE
)

from .image_generation import (
    create_bubble_prompts_file,
    generate_high_quality_bubble_image
)

from .bubble_screening import (
    BubbleScreener,
    create_bubble_screener,
    screen_bubble_images_parallel
)

from .gpu_manager import (
    GPUManager,
    get_global_gpu_manager,
    cleanup_global_gpu_manager
)

from .gpu_performance_manager import (
    GPUPerformanceManager,
    get_global_performance_manager,
    cleanup_global_performance_manager
)

from .coordinate_transform import (
    rotate_image,
    transform_3d_to_2d,
    sort_bubbles_by_depth,
    transform_3d_to_pixel_coloring_coords
)

from .flow_composition import (
    create_flow_field_composition,
    composite_bubble_to_canvas,
    load_bubble_positions,
    calculate_dynamic_canvas_size
)

from .bubble_rendering import (
    render_single_bubble,
    process_single_bubble_rendering,
    pixel_coloring,
    cv2_enhance_contrast,
    ellipsoid_fit,
    calculate_dynamic_canvas_size
)

from .flow_generator import (
    generater,
    process_projection,
    generate_uniform_points_on_sphere,
    upsample_point_cloud,
    upsample_and_scale_mesh,
    generate_points_in_cube
)

# 预生成数据集筛选系统
try:
    from .pregenerated_dataset_manager import PregeneratedDatasetManager
    from .generate_flow_field_optimized import OptimizedBubbleSelector
except ImportError:
    # 如果导入失败，提供占位符
    PregeneratedDatasetManager = None
    OptimizedBubbleSelector = None

# 定义模块的公共接口
__all__ = [
    # bubble_analysis
    'analyze_bubble_image',
    'analyze_bubble_image_from_array',
    'calculate_fourier_descriptors',
    'calculate_similarity_metrics',
    'cosine_similarity_manual',
    
    # image_generation
    'create_bubble_prompts_file',
    'generate_high_quality_bubble_image',
    
    # coordinate_transform
    'rotate_image',
    'transform_3d_to_2d',
    'sort_bubbles_by_depth',
    
    # flow_composition
    'create_flow_field_composition',
    'composite_bubble_to_canvas',
    'load_bubble_positions',
    'calculate_dynamic_canvas_size',
    
    # bubble_rendering
    'render_single_bubble',
    'process_single_bubble_rendering',
    'pixel_coloring',
    'cv2_enhance_contrast',
    'ellipsoid_fit',
    
    # flow_generator
    'generater',
    'process_projection',
    'generate_uniform_points_on_sphere',
    'upsample_point_cloud',
    'upsample_and_scale_mesh',
    'generate_points_in_cube'
]
