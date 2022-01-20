use crate::include::*;
use crate::main_vulkan::*;
use crate::buffer::*;
use crate::app::*;

// A 3D model
#[derive(Clone, Debug, Default)]
pub struct MyModel {
    // The vertices used
    pub vertices: Vec<Vertex>,

    // Triangles using the vertices(each value is an index in vertices)
    //  The triangles builds up the 3D model
    pub indices: Vec<u32>,

    // Buffer holding the vertices so that the GPU understands
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,

    // Buffer holding the indices so that the GPU understands
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
}

pub enum ModelLoader {
    ModelLoaderObjFile(ModelLoaderObjFile)
}

impl ModelLoader {
    pub fn load(&self, instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
        match self {
            ModelLoader::ModelLoaderObjFile(loader) => {
                return loader.load(instance, device, data);
            }
        }
    }
}

pub struct ModelLoaderObjFile {
    pub obj_file_path: String,
    pub mtl_file_path: String
}

impl ModelLoaderObjFile {
    pub fn new(obj_file_path: String, mtl_file_path: String) -> Self {
        Self {
            obj_file_path,
            mtl_file_path
        }
    }
    pub fn load(&self, instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {

        let mut model = MyModel::new();
        model.load_model(self.obj_file_path.as_str(), self.mtl_file_path.as_str())?;
        unsafe { model.create_buffers(data, instance, device)?; };
        data.models.push(model);

        Ok(())
    }
}

impl MyModel {
    pub fn new() -> Self {
        Self {
            vertices: Default::default(),
            indices: Default::default(),
            vertex_buffer: Default::default(),
            vertex_buffer_memory: Default::default(),
            index_buffer: Default::default(),
            index_buffer_memory: Default::default(),
        }
    }
    pub unsafe fn destroy(&mut self, device: &Device) {
        device.free_memory(self.index_buffer_memory, None);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.vertex_buffer_memory, None);
        device.destroy_buffer(self.vertex_buffer, None);
    }
    pub unsafe fn create_buffers(&mut self, data: &AppData, instance: &Instance, device: &Device) -> Result<()> {
        self.create_index_buffer(data, instance, device)?;
        self.create_vertex_buffer(data, instance, device)?;
        return Ok(());
    }
    unsafe fn create_index_buffer(&mut self, data: &AppData, instance: &Instance, device: &Device) -> Result<()> {
        // Create (staging)
    
        let size = (size_of::<u32>() * self.indices.len()) as u64;
    
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
    
        // Copy (staging)
    
        let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    
        memcpy(self.indices.as_ptr(), memory.cast(), self.indices.len());
    
        device.unmap_memory(staging_buffer_memory);
    
        // Create (index)
    
        let (index_buffer, index_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
    
        self.index_buffer = index_buffer;
        self.index_buffer_memory = index_buffer_memory;
    
        // Copy (index)
    
        copy_buffer(device, data, staging_buffer, index_buffer, size)?;
    
        // Cleanup
    
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    
        Ok(())
    }
    unsafe fn create_vertex_buffer(&mut self, data: &AppData, instance: &Instance, device: &Device) -> Result<()> {
        // Create (staging)
    
        let size = (size_of::<Vertex>() * self.vertices.len()) as u64;
    
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
    
        // Copy (staging)
    
        let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    
        memcpy(self.vertices.as_ptr(), memory.cast(), self.vertices.len());
    
        device.unmap_memory(staging_buffer_memory);
    
        // Create (vertex)
    
        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
    
        self.vertex_buffer = vertex_buffer;
        self.vertex_buffer_memory = vertex_buffer_memory;
    
        // Copy (vertex)
    
        copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;
    
        // Cleanup
    
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    
        Ok(())
    }
    // Load a model from a file
    pub fn load_model(&mut self, obj_path: &str, mtl_path: &str) -> Result<()> {
        // Load the verticies and indices, make sure there are no dubble verticies
        let mut unique_vertices = HashMap::new();

         // Read the MTL file
         let mut reader = BufReader::new(File::open(mtl_path)?);

         //let (models, materials) = tobj::load_obj(path, true)?;
 
         let materials2 = tobj::load_mtl_buf(&mut reader)?;

         let materials = (&materials2).0.clone();

        // Read the OBJ file
        let mut reader = BufReader::new(File::open(obj_path)?);

        //let (models, materials) = tobj::load_obj(path, true)?;

        let (models, _) = tobj::load_obj_buf(&mut reader, true, |_| {
            Ok(materials2.clone())
        })?;

        //let materials = materials.0;
        println!("materials: {}", materials.len());
        println!("models: {}", models.len());

        for model_index in 0..models.len() {
            let model = &models[model_index];
            println!("{}", model.name);
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let color: [f32; 3];
                if model.mesh.material_id.is_some() {
                    let material = &materials[model.mesh.material_id.unwrap()];
                    color = material.diffuse;
                }
                else {
                    color = [1.0, 0.0, 0.0];
                }

                let color: glm::TVec3<f32> = glm::vec3(color[0], color[1], color[2]);

                let vertex = Vertex {
                    pos: glm::vec3(
                        model.mesh.positions[pos_offset + 2] / RENDER_DISTANCE,
                        model.mesh.positions[pos_offset] / RENDER_DISTANCE,
                        model.mesh.positions[pos_offset + 1] / RENDER_DISTANCE,
                    ),
                    color,
                    tex_coord: glm::vec2(0.0, 0.0
                        /*model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],*/
                    ),
                };
    
                if let Some(index) = unique_vertices.get(&vertex) {
                    self.indices.push(*index as u32);
                } else {
                    let index = self.vertices.len();
                    unique_vertices.insert(vertex, index);
                    self.vertices.push(vertex);
                    self.indices.push(index as u32);
                }
            }
        }
    
        Ok(())
    }
}

// An instance of a model, there can be multiple instances of the same model
#[derive(Clone, Debug, Default)]
pub struct ModelInstance {
    pub model_index: usize,
    pub position: glm::TVec3<f32>,
    pub rotate_rad: f32,
    pub rotate_vec: glm::TVec3<f32>
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub(crate) pos: glm::Vec3,
    pub(crate) color: glm::Vec3,
    pub(crate) tex_coord: glm::Vec2,
}

impl Vertex {
    pub fn new(pos: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self { pos, color, tex_coord }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<glm::Vec3>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<glm::Vec3>() + size_of::<glm::Vec3>()) as u32)
            .build();
        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

/*pub fn load_model(instance: &Instance, device: &Device, data: &mut AppData, obj_path: &str) -> Result<()> {
    // Model

    let mut model = MyModel::new();
    model.load_model(path)?;
    unsafe { model.create_buffers(data, instance, device)?; };
    data.models.push(model);

    Ok(())
}*/