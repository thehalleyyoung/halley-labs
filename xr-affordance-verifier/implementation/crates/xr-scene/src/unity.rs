//! Unity scene format adapter.
//!
//! Provides `UnitySceneAdapter` for converting simplified Unity-style scene
//! descriptions into the internal SceneModel format.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::geometry::{BoundingBox, Capsule, Cylinder, Sphere, Volume};
use xr_types::scene::{DependencyType, InteractableElement, InteractionType, SceneMetadata, SceneModel, TransformNode};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityGameObject {
    pub file_id: u64,
    pub name: String,
    pub tag: String,
    pub layer: u32,
    pub active: bool,
    pub components: Vec<UnityComponent>,
    pub children_ids: Vec<u64>,
    pub parent_id: Option<u64>,
    pub prefab_id: Option<String>,
}

impl UnityGameObject {
    pub fn new(file_id: u64, name: impl Into<String>) -> Self {
        Self { file_id, name: name.into(), tag: "Untagged".to_string(), layer: 0,
               active: true, components: Vec::new(), children_ids: Vec::new(),
               parent_id: None, prefab_id: None }
    }
    pub fn has_component(&self, kind: ComponentKind) -> bool { self.components.iter().any(|c| c.kind == kind) }
    pub fn get_transform(&self) -> Option<&UnityTransform> {
        self.components.iter().find_map(|c| match &c.data { ComponentData::Transform(t) => Some(t), _ => None })
    }
    pub fn get_collider(&self) -> Option<&UnityCollider> {
        self.components.iter().find_map(|c| match &c.data { ComponentData::Collider(col) => Some(col), _ => None })
    }
    pub fn get_interactable(&self) -> Option<&UnityInteractable> {
        self.components.iter().find_map(|c| match &c.data { ComponentData::Interactable(i) => Some(i), _ => None })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityComponent { pub kind: ComponentKind, pub data: ComponentData }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentKind { Transform, BoxCollider, SphereCollider, CapsuleCollider, MeshCollider, Interactable, EventTrigger, Canvas, Button, Slider, Other }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentData {
    Transform(UnityTransform), Collider(UnityCollider), Interactable(UnityInteractable),
    EventTrigger(UnityEventTrigger), Button(UnityButtonData), Slider(UnitySliderData),
    Other(HashMap<String, String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityTransform {
    pub local_position: [f64; 3],
    pub local_rotation: [f64; 4],
    pub local_scale: [f64; 3],
}

impl Default for UnityTransform {
    fn default() -> Self { Self { local_position: [0.0; 3], local_rotation: [0.0, 0.0, 0.0, 1.0], local_scale: [1.0; 3] } }
}

impl UnityTransform {
    pub fn rotation_wxyz(&self) -> [f64; 4] { [self.local_rotation[3], self.local_rotation[0], self.local_rotation[1], self.local_rotation[2]] }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityCollider {
    pub collider_type: ColliderType, pub center: [f64; 3], pub size: [f64; 3],
    pub radius: f64, pub height: f64, pub direction: u32, pub is_trigger: bool,
}

impl Default for UnityCollider {
    fn default() -> Self { Self { collider_type: ColliderType::Box, center: [0.0; 3], size: [1.0; 3], radius: 0.5, height: 2.0, direction: 1, is_trigger: false } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColliderType { Box, Sphere, Capsule, Mesh }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityInteractable {
    pub interaction_type: String, pub select_mode: String,
    pub use_dynamic_attach: bool, pub throw_on_detach: bool,
    pub custom_properties: HashMap<String, String>,
}

impl Default for UnityInteractable {
    fn default() -> Self { Self { interaction_type: "XRGrabInteractable".to_string(), select_mode: "Single".to_string(), use_dynamic_attach: false, throw_on_detach: true, custom_properties: HashMap::new() } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityEventTrigger { pub event_type: String, pub callback_target: Option<u64>, pub callback_method: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityButtonData { pub interactable: bool, pub transition_type: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitySliderData { pub min_value: f64, pub max_value: f64, pub whole_numbers: bool, pub direction: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityPrefab { pub id: String, pub name: String, pub objects: Vec<UnityGameObject> }

impl UnityPrefab {
    pub fn instantiate(&self, base_id: u64, position: [f64; 3]) -> Vec<UnityGameObject> {
        let mut objects = self.objects.clone();
        let mut id_map: HashMap<u64, u64> = HashMap::new();
        for (i, obj) in objects.iter_mut().enumerate() {
            let new_id = base_id + i as u64;
            id_map.insert(obj.file_id, new_id);
            obj.file_id = new_id;
        }
        for obj in &mut objects {
            if let Some(pid) = obj.parent_id { obj.parent_id = id_map.get(&pid).copied(); }
            obj.children_ids = obj.children_ids.iter().filter_map(|id| id_map.get(id).copied()).collect();
        }
        if let Some(root) = objects.first_mut() {
            if let Some(transform) = root.components.iter_mut().find_map(|c| match &mut c.data { ComponentData::Transform(t) => Some(t), _ => None }) {
                for i in 0..3 { transform.local_position[i] += position[i]; }
            }
        }
        objects
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitySceneDescription {
    pub name: String, pub unity_version: String,
    pub objects: Vec<UnityGameObject>, pub prefabs: Vec<UnityPrefab>,
    pub event_connections: Vec<UnityEventConnection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityEventConnection { pub source_id: u64, pub target_id: u64, pub event_type: String }

pub struct UnitySceneAdapter { convert_triggers_to_proximity: bool, default_activation_radius: f64 }

impl UnitySceneAdapter {
    pub fn new() -> Self { Self { convert_triggers_to_proximity: true, default_activation_radius: 0.05 } }
    pub fn with_trigger_conversion(mut self, enabled: bool) -> Self { self.convert_triggers_to_proximity = enabled; self }
    pub fn with_activation_radius(mut self, radius: f64) -> Self { self.default_activation_radius = radius; self }

    pub fn convert(&self, unity_scene: &UnitySceneDescription) -> VerifierResult<SceneModel> {
        let mut all_objects = unity_scene.objects.clone();
        let mut next_id = all_objects.iter().map(|o| o.file_id).max().unwrap_or(0) + 1000;

        for obj in &unity_scene.objects {
            if let Some(prefab_id) = &obj.prefab_id {
                if let Some(prefab) = unity_scene.prefabs.iter().find(|p| &p.id == prefab_id) {
                    let position = obj.get_transform().map(|t| t.local_position).unwrap_or([0.0; 3]);
                    let instances = prefab.instantiate(next_id, position);
                    next_id += instances.len() as u64 + 100;
                    all_objects.extend(instances);
                }
            }
        }

        let mut scene = SceneModel::new(&unity_scene.name);
        scene.metadata = SceneMetadata { unity_version: Some(unity_scene.unity_version.clone()), ..SceneMetadata::default() };

        let mut file_id_to_transform_idx: HashMap<u64, usize> = HashMap::new();
        for obj in &all_objects {
            let transform = obj.get_transform().cloned().unwrap_or_default();
            let parent_idx = obj.parent_id.and_then(|pid| file_id_to_transform_idx.get(&pid).copied());
            let node = TransformNode {
                name: obj.name.clone(), parent: parent_idx, local_position: transform.local_position,
                local_rotation: transform.rotation_wxyz(), local_scale: transform.local_scale, children: Vec::new(),
            };
            let idx = scene.transform_nodes.len();
            scene.transform_nodes.push(node);
            file_id_to_transform_idx.insert(obj.file_id, idx);
        }

        for obj in &all_objects {
            if let Some(&idx) = file_id_to_transform_idx.get(&obj.file_id) {
                let children: Vec<usize> = obj.children_ids.iter().filter_map(|cid| file_id_to_transform_idx.get(cid).copied()).collect();
                scene.transform_nodes[idx].children = children;
            }
        }

        let mut file_id_to_element: HashMap<u64, usize> = HashMap::new();
        for obj in &all_objects {
            if !obj.active { continue; }
            if let Some(element) = self.convert_object(obj, &file_id_to_transform_idx)? {
                let idx = scene.add_element(element);
                file_id_to_element.insert(obj.file_id, idx);
            }
        }

        for conn in &unity_scene.event_connections {
            if let (Some(&src_idx), Some(&tgt_idx)) = (file_id_to_element.get(&conn.source_id), file_id_to_element.get(&conn.target_id)) {
                let dep_type = match conn.event_type.as_str() {
                    "OnClick" | "OnSelect" => DependencyType::Sequential,
                    "OnActivate" | "OnEnable" => DependencyType::Enable,
                    "OnVisible" | "OnShow" => DependencyType::Visibility,
                    "Concurrent" | "OnSimultaneous" => DependencyType::Concurrent,
                    _ => DependencyType::Sequential,
                };
                scene.add_dependency(src_idx, tgt_idx, dep_type);
            }
        }
        scene.recompute_bounds();
        Ok(scene)
    }

    fn convert_object(&self, obj: &UnityGameObject, transform_map: &HashMap<u64, usize>) -> VerifierResult<Option<InteractableElement>> {
        let interactable = match obj.get_interactable() {
            Some(i) => i,
            None => {
                let has_button = obj.has_component(ComponentKind::Button);
                let has_slider = obj.has_component(ComponentKind::Slider);
                if !has_button && !has_slider {
                    if self.convert_triggers_to_proximity {
                        if let Some(collider) = obj.get_collider() {
                            if collider.is_trigger {
                                return Ok(Some(self.make_proximity_element(obj, collider, transform_map)));
                            }
                        }
                    }
                    return Ok(None);
                }
                &UnityInteractable::default()
            }
        };
        let transform = obj.get_transform().cloned().unwrap_or_default();
        let position = transform.local_position;
        let interaction_type = self.map_interaction_type(interactable, obj);
        let volume = self.build_volume(obj, &transform);
        let mut element = InteractableElement::new(&obj.name, position, interaction_type);
        element.orientation = transform.rotation_wxyz();
        element.scale = transform.local_scale;
        element.activation_volume = volume;
        element.transform_node = transform_map.get(&obj.file_id).copied();
        if obj.tag != "Untagged" { element.tags.push(obj.tag.clone()); }
        element.properties.insert("unity_file_id".to_string(), obj.file_id.to_string());
        Ok(Some(element))
    }

    fn make_proximity_element(&self, obj: &UnityGameObject, collider: &UnityCollider, transform_map: &HashMap<u64, usize>) -> InteractableElement {
        let transform = obj.get_transform().cloned().unwrap_or_default();
        let volume = self.collider_to_volume(collider, &transform);
        let mut element = InteractableElement::new(&obj.name, transform.local_position, InteractionType::Proximity);
        element.activation_volume = volume;
        element.transform_node = transform_map.get(&obj.file_id).copied();
        element
    }

    fn map_interaction_type(&self, interactable: &UnityInteractable, obj: &UnityGameObject) -> InteractionType {
        if obj.has_component(ComponentKind::Button) { return InteractionType::Click; }
        if obj.has_component(ComponentKind::Slider) { return InteractionType::Slider; }
        match interactable.interaction_type.as_str() {
            "XRGrabInteractable" | "GrabInteractable" => InteractionType::Grab,
            "XRSimpleInteractable" | "SimpleInteractable" => InteractionType::Click,
            "XRSocketInteractable" | "SocketInteractable" => InteractionType::Drag,
            "XRPokeInteractable" | "PokeInteractable" => InteractionType::Click,
            "XRGazeInteractable" | "GazeInteractable" => InteractionType::Gaze,
            _ => InteractionType::Click,
        }
    }

    fn build_volume(&self, obj: &UnityGameObject, transform: &UnityTransform) -> Volume {
        if let Some(collider) = obj.get_collider() { self.collider_to_volume(collider, transform) }
        else { Volume::Sphere(Sphere::new(transform.local_position, self.default_activation_radius)) }
    }

    fn collider_to_volume(&self, collider: &UnityCollider, transform: &UnityTransform) -> Volume {
        let pos = transform.local_position;
        let scale = transform.local_scale;
        let center = [pos[0] + collider.center[0] * scale[0], pos[1] + collider.center[1] * scale[1], pos[2] + collider.center[2] * scale[2]];
        match collider.collider_type {
            ColliderType::Box => {
                let half = [collider.size[0] * scale[0] * 0.5, collider.size[1] * scale[1] * 0.5, collider.size[2] * scale[2] * 0.5];
                Volume::Box(BoundingBox::from_center_extents(center, half))
            }
            ColliderType::Sphere => {
                let max_scale = scale[0].max(scale[1]).max(scale[2]);
                Volume::Sphere(Sphere::new(center, collider.radius * max_scale))
            }
            ColliderType::Capsule => {
                let max_scale = scale[0].max(scale[2]);
                let radius = collider.radius * max_scale;
                let half_h = (collider.height * scale[1] * 0.5 - radius).max(0.0);
                let axis = match collider.direction { 0 => [1.0, 0.0, 0.0], 2 => [0.0, 0.0, 1.0], _ => [0.0, 1.0, 0.0] };
                let start = [center[0] - axis[0] * half_h, center[1] - axis[1] * half_h, center[2] - axis[2] * half_h];
                let end = [center[0] + axis[0] * half_h, center[1] + axis[1] * half_h, center[2] + axis[2] * half_h];
                Volume::Capsule(Capsule::new(start, end, radius))
            }
            ColliderType::Mesh => {
                let half = [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5];
                Volume::Box(BoundingBox::from_center_extents(center, half))
            }
        }
    }

    pub fn parse_json(&self, json_str: &str) -> VerifierResult<UnitySceneDescription> {
        serde_json::from_str(json_str).map_err(|e| VerifierError::SceneParsing(format!("Unity JSON parse error: {}", e)))
    }

    pub fn parse_and_convert(&self, json_str: &str) -> VerifierResult<SceneModel> {
        let desc = self.parse_json(json_str)?;
        self.convert(&desc)
    }
}

impl Default for UnitySceneAdapter { fn default() -> Self { Self::new() } }

pub fn make_interactable_object(file_id: u64, name: &str, position: [f64; 3], collider_type: ColliderType, interaction: &str) -> UnityGameObject {
    let mut obj = UnityGameObject::new(file_id, name);
    obj.components.push(UnityComponent { kind: ComponentKind::Transform, data: ComponentData::Transform(UnityTransform { local_position: position, local_rotation: [0.0, 0.0, 0.0, 1.0], local_scale: [1.0; 3] }) });
    let collider = UnityCollider { collider_type, center: [0.0; 3], size: [0.1; 3], radius: 0.05, height: 0.2, direction: 1, is_trigger: false };
    let comp_kind = match collider_type { ColliderType::Box => ComponentKind::BoxCollider, ColliderType::Sphere => ComponentKind::SphereCollider, ColliderType::Capsule => ComponentKind::CapsuleCollider, ColliderType::Mesh => ComponentKind::MeshCollider };
    obj.components.push(UnityComponent { kind: comp_kind, data: ComponentData::Collider(collider) });
    obj.components.push(UnityComponent { kind: ComponentKind::Interactable, data: ComponentData::Interactable(UnityInteractable { interaction_type: interaction.to_string(), ..UnityInteractable::default() }) });
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unity_scene() -> UnitySceneDescription {
        let button = make_interactable_object(1, "Button1", [0.0, 1.0, -0.5], ColliderType::Box, "XRSimpleInteractable");
        let handle = make_interactable_object(2, "Handle1", [0.3, 1.2, -0.5], ColliderType::Sphere, "XRGrabInteractable");
        UnitySceneDescription {
            name: "TestUnityScene".to_string(), unity_version: "2022.3".to_string(),
            objects: vec![button, handle], prefabs: Vec::new(),
            event_connections: vec![UnityEventConnection { source_id: 1, target_id: 2, event_type: "OnClick".to_string() }],
        }
    }

    #[test] fn test_convert_basic_scene() {
        let unity = make_unity_scene();
        let adapter = UnitySceneAdapter::new();
        let scene = adapter.convert(&unity).unwrap();
        assert_eq!(scene.name, "TestUnityScene");
        assert_eq!(scene.elements.len(), 2);
        assert_eq!(scene.dependencies.len(), 1);
        assert_eq!(scene.elements[0].interaction_type, InteractionType::Click);
        assert_eq!(scene.elements[1].interaction_type, InteractionType::Grab);
    }

    #[test] fn test_transform_conversion() {
        let unity = make_unity_scene();
        let adapter = UnitySceneAdapter::new();
        let scene = adapter.convert(&unity).unwrap();
        assert!((scene.elements[0].position[1] - 1.0).abs() < 1e-10);
    }

    #[test] fn test_collider_to_volume_box() {
        let adapter = UnitySceneAdapter::new();
        let collider = UnityCollider { collider_type: ColliderType::Box, center: [0.0; 3], size: [0.2, 0.1, 0.05], ..UnityCollider::default() };
        let transform = UnityTransform::default();
        let vol = adapter.collider_to_volume(&collider, &transform);
        match vol { Volume::Box(bb) => { assert!((bb.extents()[0] - 0.2).abs() < 1e-10); } _ => panic!("Expected box") }
    }

    #[test] fn test_collider_to_volume_sphere() {
        let adapter = UnitySceneAdapter::new();
        let collider = UnityCollider { collider_type: ColliderType::Sphere, radius: 0.1, ..UnityCollider::default() };
        let vol = adapter.collider_to_volume(&collider, &UnityTransform::default());
        match vol { Volume::Sphere(s) => assert!((s.radius - 0.1).abs() < 1e-10), _ => panic!("Expected sphere") }
    }

    #[test] fn test_trigger_to_proximity() {
        let mut obj = UnityGameObject::new(1, "TriggerZone");
        obj.components.push(UnityComponent { kind: ComponentKind::Transform, data: ComponentData::Transform(UnityTransform::default()) });
        obj.components.push(UnityComponent { kind: ComponentKind::BoxCollider, data: ComponentData::Collider(UnityCollider { collider_type: ColliderType::Box, is_trigger: true, size: [2.0; 3], ..UnityCollider::default() }) });
        let scene_desc = UnitySceneDescription { name: "TriggerTest".to_string(), unity_version: "2022.3".to_string(), objects: vec![obj], prefabs: Vec::new(), event_connections: Vec::new() };
        let scene = UnitySceneAdapter::new().convert(&scene_desc).unwrap();
        assert_eq!(scene.elements.len(), 1);
        assert_eq!(scene.elements[0].interaction_type, InteractionType::Proximity);
    }

    #[test] fn test_inactive_objects_excluded() {
        let mut obj = make_interactable_object(1, "Inactive", [0.0; 3], ColliderType::Box, "XRSimpleInteractable");
        obj.active = false;
        let scene_desc = UnitySceneDescription { name: "InactiveTest".to_string(), unity_version: "2022.3".to_string(), objects: vec![obj], prefabs: Vec::new(), event_connections: Vec::new() };
        let scene = UnitySceneAdapter::new().convert(&scene_desc).unwrap();
        assert_eq!(scene.elements.len(), 0);
    }

    #[test] fn test_rotation_conversion() {
        let t = UnityTransform { local_position: [0.0; 3], local_rotation: [0.1, 0.2, 0.3, 0.9], local_scale: [1.0; 3] };
        let wxyz = t.rotation_wxyz();
        assert!((wxyz[0] - 0.9).abs() < 1e-10);
        assert!((wxyz[1] - 0.1).abs() < 1e-10);
    }

    #[test] fn test_event_type_mapping() {
        let mut scene = make_unity_scene();
        scene.event_connections = vec![UnityEventConnection { source_id: 1, target_id: 2, event_type: "OnEnable".to_string() }];
        let model = UnitySceneAdapter::new().convert(&scene).unwrap();
        assert_eq!(model.dependencies[0].dependency_type, DependencyType::Enable);
    }

    #[test] fn test_parse_json() {
        let json = serde_json::to_string(&make_unity_scene()).unwrap();
        let adapter = UnitySceneAdapter::new();
        let desc = adapter.parse_json(&json).unwrap();
        assert_eq!(desc.name, "TestUnityScene");
    }

    #[test] fn test_prefab_expansion() {
        let prefab_obj = make_interactable_object(100, "PrefabButton", [0.0; 3], ColliderType::Box, "XRSimpleInteractable");
        let prefab = UnityPrefab { id: "prefab_btn".to_string(), name: "ButtonPrefab".to_string(), objects: vec![prefab_obj] };
        let mut root = UnityGameObject::new(1, "Root");
        root.components.push(UnityComponent { kind: ComponentKind::Transform, data: ComponentData::Transform(UnityTransform { local_position: [1.0, 2.0, 3.0], ..UnityTransform::default() }) });
        root.prefab_id = Some("prefab_btn".to_string());
        let scene_desc = UnitySceneDescription { name: "PrefabTest".to_string(), unity_version: "2022.3".to_string(), objects: vec![root], prefabs: vec![prefab], event_connections: Vec::new() };
        let scene = UnitySceneAdapter::new().convert(&scene_desc).unwrap();
        assert!(scene.elements.len() >= 1);
    }
}
