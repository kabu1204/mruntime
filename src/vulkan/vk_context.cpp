#include "vk_context.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "vk_helpers.h"

namespace mruntime::vulkan {

namespace {

constexpr uint32_t kTimestampQueryCount = 4096;

constexpr const char* kKhrPortabilitySubsetExtensionName = "VK_KHR_portability_subset";
constexpr VkSubgroupFeatureFlags kRequiredSubgroupOperations =
    VK_SUBGROUP_FEATURE_BASIC_BIT |
    VK_SUBGROUP_FEATURE_VOTE_BIT |
    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
    VK_SUBGROUP_FEATURE_BALLOT_BIT |
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
    VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
    VK_SUBGROUP_FEATURE_QUAD_BIT;

void require_fp16_features(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16 = {};
    float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;

    VkPhysicalDevice16BitStorageFeatures storage16 = {};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    float16.pNext = &storage16;

    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_layout = {};
    scalar_layout.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT;
    storage16.pNext = &scalar_layout;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &float16;

    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    if (!float16.shaderFloat16) {
        throw std::runtime_error("Required Vulkan feature missing: shaderFloat16");
    }
    if (!storage16.storageBuffer16BitAccess) {
        throw std::runtime_error("Required Vulkan feature missing: storageBuffer16BitAccess");
    }
    if (!scalar_layout.scalarBlockLayout) {
        throw std::runtime_error("Required Vulkan feature missing: scalarBlockLayout");
    }
}

void require_vulkan13_features(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features13;

    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    if (!features13.synchronization2) {
        throw std::runtime_error("Required Vulkan 1.3 feature missing: synchronization2");
    }
    if (!features13.maintenance4) {
        throw std::runtime_error("Required Vulkan 1.3 feature missing: maintenance4");
    }
    if (!features13.computeFullSubgroups) {
        throw std::runtime_error("Required Vulkan 1.3 feature missing: computeFullSubgroups");
    }
}

void require_subgroup_support(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceSubgroupProperties subgroup_props = {};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);

    if (subgroup_props.subgroupSize == 0) {
        throw std::runtime_error("Required Vulkan capability missing: subgroup support");
    }
    if ((subgroup_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        throw std::runtime_error("Required Vulkan capability missing: compute-stage subgroup support");
    }
    if ((subgroup_props.supportedOperations & kRequiredSubgroupOperations) !=
        kRequiredSubgroupOperations) {
        throw std::runtime_error(
            "Required Vulkan subgroup operations missing: basic/vote/arithmetic/ballot/"
            "shuffle/shuffle-relative/quad");
    }
}

bool device_supports_required_extensions(VkPhysicalDevice physical_device, bool* out_has_portability_subset) {
    const auto exts = enumerate_device_extensions(physical_device);
    const std::array<const char*, 3> required = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
    };
    for (const char* ext : required) {
        if (!has_extension(exts, ext)) {
            return false;
        }
    }
    if (out_has_portability_subset) {
        *out_has_portability_subset = has_extension(exts, kKhrPortabilitySubsetExtensionName);
    }
    return true;
}

bool has_time_domain(const std::vector<VkTimeDomainKHR>& domains, VkTimeDomainKHR domain) {
    return std::find(domains.begin(), domains.end(), domain) != domains.end();
}

}  // namespace

VkContext VkContext::Create(const VkContextCreateInfo& info) {
    VkContext ctx;

    const auto instance_exts = enumerate_instance_extensions();
    std::vector<const char*> enabled_instance_exts;
    std::vector<const char*> enabled_layers;
    VkInstanceCreateFlags instance_flags = 0;

    if (has_extension(instance_exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabled_instance_exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    // Validation layers: enable when requested and the layer is installed.
    VkValidationFeaturesEXT validation_features = {};
    std::array<VkValidationFeatureEnableEXT, 2> validation_enables = {
        VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };

    bool use_validation = false;
    if (info.enable_validation) {
        const auto layers = enumerate_instance_layers();
        if (has_layer(layers, "VK_LAYER_KHRONOS_validation")) {
            enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
            use_validation = true;
        }
    }

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "mruntime";
    app.applicationVersion = 1;
    app.pEngineName = "mruntime";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_ci = {};
    instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_ci.flags = instance_flags;
    instance_ci.pApplicationInfo = &app;
    instance_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_instance_exts.size());
    instance_ci.ppEnabledExtensionNames =
        enabled_instance_exts.empty() ? nullptr : enabled_instance_exts.data();
    instance_ci.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    instance_ci.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();

    if (use_validation) {
        validation_features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validation_features.enabledValidationFeatureCount =
            static_cast<uint32_t>(validation_enables.size());
        validation_features.pEnabledValidationFeatures = validation_enables.data();
        instance_ci.pNext = &validation_features;
    }

    vk_check(vkCreateInstance(&instance_ci, nullptr, &ctx.instance_), "vkCreateInstance");

    uint32_t device_count = 0;
    vk_check(vkEnumeratePhysicalDevices(ctx.instance_, &device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vk_check(vkEnumeratePhysicalDevices(ctx.instance_, &device_count, devices.data()), "vkEnumeratePhysicalDevices");

    bool selected_has_portability_subset = false;
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties dev_props = {};
        vkGetPhysicalDeviceProperties(dev, &dev_props);
        if (VK_API_VERSION_MAJOR(dev_props.apiVersion) < 1 ||
            (VK_API_VERSION_MAJOR(dev_props.apiVersion) == 1 &&
             VK_API_VERSION_MINOR(dev_props.apiVersion) < 3)) {
            continue;
        }

        bool has_portability_subset = false;
        if (!device_supports_required_extensions(dev, &has_portability_subset)) {
            continue;
        }

        try {
            (void)find_compute_queue_family(dev);
            require_fp16_features(dev);
            require_vulkan13_features(dev);
            require_subgroup_support(dev);
        } catch (...) {
            continue;
        }

        ctx.physical_device_ = dev;
        selected_has_portability_subset = has_portability_subset;
        break;
    }

    if (ctx.physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No Vulkan 1.3+ physical device satisfies required extensions/features");
    }

    const uint32_t queue_family_index = find_compute_queue_family(ctx.physical_device_);
    ctx.queue_family_index_ = queue_family_index;

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(ctx.physical_device_, &props);
    ctx.min_storage_buffer_offset_alignment_ = props.limits.minStorageBufferOffsetAlignment;
    ctx.timestamp_period_ns_ = props.limits.timestampPeriod;

    // Timestamp queries: check support on the selected queue family.
    {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device_, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device_, &qf_count, qf_props.data());
        if (queue_family_index < qf_count) {
            ctx.timestamp_valid_bits_ = qf_props[queue_family_index].timestampValidBits;
        }
    }

    // Optional calibrated timestamps extension (prefer KHR, fall back to EXT alias).
    const auto dev_exts = enumerate_device_extensions(ctx.physical_device_);
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    ctx.cooperative_matrix_supported_ =
        has_extension(dev_exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif
    const char* calibrated_ext_name = nullptr;
#if defined(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
    if (has_extension(dev_exts, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        calibrated_ext_name = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
    }
#endif
#if defined(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
    if (calibrated_ext_name == nullptr &&
        has_extension(dev_exts, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        calibrated_ext_name = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
    }
#endif

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo q_ci = {};
    q_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q_ci.queueFamilyIndex = queue_family_index;
    q_ci.queueCount = 1;
    q_ci.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16 = {};
    float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
    float16.shaderFloat16 = VK_TRUE;

    VkPhysicalDevice16BitStorageFeatures storage16 = {};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16.storageBuffer16BitAccess = VK_TRUE;
    float16.pNext = &storage16;

    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_layout = {};
    scalar_layout.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT;
    scalar_layout.scalarBlockLayout = VK_TRUE;
    storage16.pNext = &scalar_layout;

    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.maintenance4 = VK_TRUE;
    features13.computeFullSubgroups = VK_TRUE;
    scalar_layout.pNext = &features13;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &float16;

    std::vector<const char*> enabled_dev_exts = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
    };
    if (selected_has_portability_subset) {
        enabled_dev_exts.push_back(kKhrPortabilitySubsetExtensionName);
    }
    if (calibrated_ext_name != nullptr) {
        enabled_dev_exts.push_back(calibrated_ext_name);
    }

    VkDeviceCreateInfo dev_ci = {};
    dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.pNext = &features2;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &q_ci;
    dev_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_dev_exts.size());
    dev_ci.ppEnabledExtensionNames = enabled_dev_exts.data();

    vk_check(vkCreateDevice(ctx.physical_device_, &dev_ci, nullptr, &ctx.device_), "vkCreateDevice");

    // Timestamp query pool: optional.
    if (ctx.timestamp_period_ns_ > 0.0f && ctx.timestamp_valid_bits_ > 0) {
        VkQueryPoolCreateInfo qp_ci = {};
        qp_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp_ci.queryCount = kTimestampQueryCount;
        if (vkCreateQueryPool(ctx.device_, &qp_ci, nullptr, &ctx.timestamp_query_pool_) != VK_SUCCESS) {
            ctx.timestamp_query_pool_ = VK_NULL_HANDLE;
        } else {
            ctx.timestamp_query_count_ = kTimestampQueryCount;
        }
    }

    VkPipelineCacheCreateInfo cache_ci = {};
    cache_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vk_check(vkCreatePipelineCache(ctx.device_, &cache_ci, nullptr, &ctx.pipeline_cache_), "vkCreatePipelineCache");

    vkGetDeviceQueue(ctx.device_, queue_family_index, 0, &ctx.queue_);

    VkCommandPoolCreateInfo pool_ci = {};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = queue_family_index;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_check(vkCreateCommandPool(ctx.device_, &pool_ci, nullptr, &ctx.command_pool_), "vkCreateCommandPool");

    // Persistent command buffer (reset per dispatch, never freed individually).
    VkCommandBufferAllocateInfo cb_ai = {};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx.command_pool_;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(ctx.device_, &cb_ai, &ctx.command_buffer_), "vkAllocateCommandBuffers");

    // Persistent fence (reset per dispatch, never destroyed individually).
    VkFenceCreateInfo fence_ci = {};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(ctx.device_, &fence_ci, nullptr, &ctx.fence_), "vkCreateFence");

    // Optional calibrated timestamps support (best-effort).
    if (calibrated_ext_name != nullptr) {
        ctx.get_time_domains_ =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                vkGetInstanceProcAddr(ctx.instance_, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"));
        if (ctx.get_time_domains_ == nullptr) {
            ctx.get_time_domains_ =
                reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                    vkGetInstanceProcAddr(ctx.instance_, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
        }

        ctx.get_calibrated_timestamps_ =
            reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                vkGetDeviceProcAddr(ctx.device_, "vkGetCalibratedTimestampsKHR"));
        if (ctx.get_calibrated_timestamps_ == nullptr) {
            ctx.get_calibrated_timestamps_ =
                reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                    vkGetDeviceProcAddr(ctx.device_, "vkGetCalibratedTimestampsEXT"));
        }

        if (ctx.get_time_domains_ != nullptr && ctx.get_calibrated_timestamps_ != nullptr) {
            uint32_t domain_count = 0;
            if (ctx.get_time_domains_(ctx.physical_device_, &domain_count, nullptr) == VK_SUCCESS &&
                domain_count > 0) {
                std::vector<VkTimeDomainKHR> domains(domain_count);
                if (ctx.get_time_domains_(ctx.physical_device_, &domain_count, domains.data()) == VK_SUCCESS) {
                    VkTimeDomainKHR host_domain = static_cast<VkTimeDomainKHR>(0);

                    // Prefer CLOCK_MONOTONIC, which most closely matches std::chrono::steady_clock on common platforms.
                    if (has_time_domain(domains, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR)) {
                        host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
                    } else if (has_time_domain(domains, VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR)) {
                        host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
                    }

                    if (host_domain != static_cast<VkTimeDomainKHR>(0)) {
                        ctx.calibrated_host_domain_ = host_domain;
                        ctx.calibrated_host_domain_is_ns_ = true;
                    }
                }
            }
        }
    }

    return ctx;
}

VkContext::~VkContext() {
    reset();
}

VkContext::VkContext(VkContext&& other) noexcept {
    *this = std::move(other);
}

VkContext& VkContext::operator=(VkContext&& other) noexcept {
    if (this == &other) return *this;
    reset();

    instance_ = std::exchange(other.instance_, VK_NULL_HANDLE);
    physical_device_ = std::exchange(other.physical_device_, VK_NULL_HANDLE);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    pipeline_cache_ = std::exchange(other.pipeline_cache_, VK_NULL_HANDLE);
    queue_ = std::exchange(other.queue_, VK_NULL_HANDLE);
    queue_family_index_ = std::exchange(other.queue_family_index_, UINT32_MAX);
    command_pool_ = std::exchange(other.command_pool_, VK_NULL_HANDLE);
    command_buffer_ = std::exchange(other.command_buffer_, VK_NULL_HANDLE);
    fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
    timestamp_query_pool_ = std::exchange(other.timestamp_query_pool_, VK_NULL_HANDLE);
    timestamp_query_count_ = std::exchange(other.timestamp_query_count_, 0u);
    min_storage_buffer_offset_alignment_ = std::exchange(other.min_storage_buffer_offset_alignment_, VkDeviceSize{0});
    timestamp_period_ns_ = std::exchange(other.timestamp_period_ns_, 0.0f);
    timestamp_valid_bits_ = std::exchange(other.timestamp_valid_bits_, 0u);
    cooperative_matrix_supported_ = std::exchange(other.cooperative_matrix_supported_, false);

    get_time_domains_ = std::exchange(other.get_time_domains_, nullptr);
    get_calibrated_timestamps_ = std::exchange(other.get_calibrated_timestamps_, nullptr);
    calibrated_host_domain_ = std::exchange(other.calibrated_host_domain_, static_cast<VkTimeDomainKHR>(0));
    calibrated_host_domain_is_ns_ = std::exchange(other.calibrated_host_domain_is_ns_, false);

    return *this;
}

void VkContext::reset() noexcept {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);

        if (timestamp_query_pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, timestamp_query_pool_, nullptr);
            timestamp_query_pool_ = VK_NULL_HANDLE;
        }
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        command_buffer_ = VK_NULL_HANDLE;
        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        if (pipeline_cache_ != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
            pipeline_cache_ = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    pipeline_cache_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_index_ = UINT32_MAX;
    physical_device_ = VK_NULL_HANDLE;
    timestamp_query_count_ = 0u;
    timestamp_period_ns_ = 0.0f;
    timestamp_valid_bits_ = 0u;
    cooperative_matrix_supported_ = false;

    get_time_domains_ = nullptr;
    get_calibrated_timestamps_ = nullptr;
    calibrated_host_domain_ = static_cast<VkTimeDomainKHR>(0);
    calibrated_host_domain_is_ns_ = false;

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VkContext::supports_calibrated_timestamps() const noexcept {
    return device_ != VK_NULL_HANDLE &&
           get_time_domains_ != nullptr &&
           get_calibrated_timestamps_ != nullptr &&
           calibrated_host_domain_is_ns_;
}

bool VkContext::calibrated_timestamps_sample(uint64_t* out_device_ticks, uint64_t* out_host_ns, uint64_t* out_max_dev_ns) const {
    if (out_device_ticks == nullptr || out_host_ns == nullptr || out_max_dev_ns == nullptr) {
        throw std::runtime_error("VkContext::calibrated_timestamps_sample: output pointer is null");
    }
    if (!supports_calibrated_timestamps()) {
        return false;
    }

    const VkTimeDomainKHR device_domain = VK_TIME_DOMAIN_DEVICE_KHR;

    VkCalibratedTimestampInfoKHR infos[2] = {};
    infos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
    infos[0].timeDomain = device_domain;
    infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
    infos[1].timeDomain = calibrated_host_domain_;

    uint64_t timestamps[2] = {};
    uint64_t max_dev_ns = 0;
    VkResult result = get_calibrated_timestamps_(device_, 2, infos, timestamps, &max_dev_ns);
    vk_check(result, "vkGetCalibratedTimestampsKHR/EXT");

    *out_device_ticks = timestamps[0];
    *out_host_ns = timestamps[1];
    *out_max_dev_ns = max_dev_ns;
    return true;
}

}  // namespace mruntime::vulkan
