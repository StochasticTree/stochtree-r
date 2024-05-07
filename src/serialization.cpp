#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <nlohmann/json.hpp>
#include <functional>
#include <memory>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<nlohmann::json> init_json_cpp() {
    std::unique_ptr<nlohmann::json> json_ptr = std::make_unique<nlohmann::json>();
    json forests = nlohmann::json::object();
    json rfx = nlohmann::json::object();
    json parameters = nlohmann::json::object();
    json_ptr->emplace("forests", forests);
    json_ptr->emplace("random_effects", rfx);
    json_ptr->emplace("parameters", parameters);
    json_ptr->emplace("num_forests", 0);
    json_ptr->emplace("num_random_effects", 0);
    json_ptr->emplace("num_parameters", 0);
    return cpp11::external_pointer<nlohmann::json>(json_ptr.release());
}

[[cpp11::register]]
void json_add_forest_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    int forest_num = json_ptr->at("num_forests");
    std::string forest_label = "forest_" + std::to_string(forest_num);
    nlohmann::json forest_json = forest_samples->to_json();
    json_ptr->at("forests").emplace(forest_label, forest_json);
}

[[cpp11::register]]
void json_add_rfx_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_samples) {
    int rfx_num = json_ptr->at("num_random_effects");
    std::string rfx_label = "random_effect_" + std::to_string(rfx_num);
    nlohmann::json rfx_json = rfx_samples->to_json();
    json_ptr->at("random_effects").emplace(rfx_label, rfx_json);
}

[[cpp11::register]]
void json_save_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string filename) {
    std::ofstream output_file(filename);
    output_file << *json_ptr << std::endl;
}
