#include "WebGPURFRender.h"
#include <cstdio>
#include <cinttypes>
#include <GLFW/glfw3.h>
#include <webgpu/webgpu_cpp.h>
#include <vector>
#include <functional>
#include <string>
#if defined(__EMSCRIPTEN__)
#include <include/emscripten.h>
#else
#include <webgpu/webgpu_glfw.h>
#endif

wgpu::Instance mInstance;
wgpu::Device mDevice;
wgpu::SwapChain mSwapChain;
wgpu::RenderPipeline mPipeline;
wgpu::CommandEncoder mEncoder;
wgpu::RenderPassEncoder mPassEncoder;
wgpu::Sampler mSampler0;
wgpu::Sampler mSampler1;
wgpu::Sampler mSampler2;
wgpu::Buffer outputBuffer;
wgpu::Buffer stagingBuffer;

CacheBuf buf = {0};
YUVBuf yuvBuf = { 0 };

const uint32_t width = 640;
const uint32_t height = 480;
bool isStartRendering = false;

const char* SHADERS = R"(
	struct VertexOutput {
      @builtin(position) Position: vec4<f32>,
      @location(0) uv: vec2<f32>,
    }

    @vertex
    fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
      var pos = array<vec2<f32>, 6>(
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0)
      );

      var uv = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
      );

      var output : VertexOutput;
      output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
      output.uv = uv[VertexIndex];
      return output;
    }


    @group(0) @binding(0) var sampler0 : sampler;
    @group(0) @binding(1) var yTex : texture_2d<f32>;
    @group(0) @binding(2) var uTex : texture_2d<f32>;
    @group(0) @binding(3) var vTex : texture_2d<f32>;
    @group(0) @binding(4) var<storage, read_write> outputBuffer : array<f32>;

    @fragment
    fn frag_main(@location(0) texCoord : vec2f) -> @location(0) vec4f {
        let y = textureSample(yTex, sampler0, texCoord).r;
        let u = textureSample(uTex, sampler0, texCoord).r;
        let v = textureSample(vTex, sampler0, texCoord).r;
        
        let yuv_2_rgb_matrix = mat4x4(
          1.1643835616, 0, 1.7927410714, -0.9729450750,
          1.1643835616, -0.2132486143, -0.5329093286, 0.3014826655,
          1.1643835616, 2.1124017857, 0, -1.1334022179,
          0, 0, 0, 0);

        let color: vec4<f32> = vec4<f32>(y, u, v, 1.0) * yuv_2_rgb_matrix;
        outputBuffer[0] = y;
        outputBuffer[1] = u;
        outputBuffer[2] = v;
        outputBuffer[3] = color.r;
        outputBuffer[4] = color.g;
        outputBuffer[5] = color.b;
        outputBuffer[6] = color.a;

        return color;
    }
)";



const char* YUV_SHADERS = R"(
	struct VertexOutput {
      @builtin(position) Position: vec4<f32>,
      @location(0) uv: vec2<f32>,
    }

    @vertex
    fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
      var pos = array<vec2<f32>, 6>(
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0)
      );

      var uv = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
      );

      var output : VertexOutput;
      output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
      output.uv = uv[VertexIndex];
      return output;
    }


    @group(0) @binding(0) var sampler0 : sampler;
    @group(0) @binding(1) var yuvTexture : texture_2d<f32>;

    @fragment
    fn frag_main(@location(0) texCoord : vec2f) -> @location(0) vec4f {
       return textureSample(yuvTexture, sampler0, texCoord);
    }
)";

WebGPURFRender::WebGPURFRender()
{
    
}

WebGPURFRender::~WebGPURFRender()
{

}

void WebGPURFRender::OnBufferReceived(uint8_t* ybuf, size_t ylen, uint8_t* ubuf, size_t ulen, uint8_t* vbuf, size_t vlen)
{
    if (!isStartRendering) isStartRendering = true;
    // CacheTextureBuffers(ybuf, ylen, uvbuf, uvlen);
    buf.ybuf = ybuf;
    buf.ylen = ylen;
    buf.ubuf = ubuf;
    buf.ulen = ulen;
    buf.vbuf = vbuf;
    buf.vlen = vlen;
}

void WebGPURFRender::OnYUVBufferReceived(uint8_t* yuvbuf, size_t len)
{
    if (!isStartRendering) isStartRendering = true;
    yuvBuf.yuvbuf = yuvbuf;
    yuvBuf.len = len;
    yuvBuf.ylen = width * height;
    yuvBuf.ulen = width * height / 4;
    yuvBuf.vlen = width * height / 4;
}

void WebGPURFRender::CreateExternalTexture()
{
    size_t ySize = width * height;
    size_t uvSize = width * height / 2;
    size_t yOffset = 0;
    size_t uvOffset = ySize;
    

    uint8_t* yPlaneData = new uint8_t[ySize];
    uint8_t* uvPlaneData = new uint8_t[uvSize];

    memcpy(yPlaneData, yuvBuf.yuvbuf + yOffset, ySize);
    memcpy(uvPlaneData, yuvBuf.yuvbuf + uvOffset, uvSize);

    wgpu::Extent3D yTexSize
    {
        .width = width,
        .height = height,
        .depthOrArrayLayers = 1
    };

    wgpu::Extent3D uvTexSize
    {
        .width = width / 2,
        .height = height / 2,
        .depthOrArrayLayers = 1
    };

    wgpu::TextureDescriptor yTexDesc
    {
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = yTexSize,
        .format = wgpu::TextureFormat::R8Unorm,
    };

    wgpu::TextureDescriptor uvTexDesc
    {
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = uvTexSize,
        .format = wgpu::TextureFormat::RG8Unorm,
    };

    wgpu::Texture yTex = mDevice.CreateTexture(&yTexDesc);
    wgpu::Texture uvTex = mDevice.CreateTexture(&uvTexDesc);

    wgpu::ImageCopyTexture yPlaneImgTex
    {
        .texture = yTex,
    };

    wgpu::ImageCopyTexture uvPlaneImgTex
    {
        .texture = uvTex,
    };

    wgpu::TextureDataLayout yTexDataLayout
    {
        .offset = 0,
        .bytesPerRow = width
    };

    wgpu::TextureDataLayout uvTexDataLayout
    {
        .offset = 0,
        .bytesPerRow = width / 2
    };

    mDevice.GetQueue().WriteTexture(&yPlaneImgTex, yPlaneData, ySize, &yTexDataLayout, &yTexDesc.size);
    mDevice.GetQueue().WriteTexture(&uvPlaneImgTex, uvPlaneData, uvSize, &uvTexDataLayout, &uvTexDesc.size);

    /*wgpu::ExternalTextureDescriptor exTexDesc
    {
        .plane0 = yTex.CreateView(),
        .plane1 = uvTex.CreateView(),
    };*/
}

void WebGPURFRender::Prepare()
{
    mInstance = wgpu::CreateInstance();
    std::function<void(wgpu::Device)> callback = [&](wgpu::Device device) {
        LogToConsole("GetDevice is called");
        mDevice = device;
        Start();
    };

    GetDevice(callback);
}

void WebGPURFRender::Start()
{
    wgpu::SurfaceDescriptorFromCanvasHTMLSelector canvasDesc{};
    canvasDesc.selector = "#canvas";

    wgpu::SurfaceDescriptor surfaceDesc{ .nextInChain = &canvasDesc };
    wgpu::Surface surface = mInstance.CreateSurface(&surfaceDesc);
    InitGraphics(surface);
}

void WebGPURFRender::StartRendering()
{
#if defined(__EMSCRIPTEN__)
    emscripten_set_main_loop_arg(WebGPURFRender::RenderWrapper, this, 0, false);
#endif
}

void WebGPURFRender::GetDevice(std::function<void(wgpu::Device)> callback)
{
    mInstance.RequestAdapter(
        nullptr,
        [](WGPURequestAdapterStatus status, WGPUAdapter wAdapter, const char* msg, void* userdata)
        {
            if (status != WGPURequestAdapterStatus_Success)
            {
                exit(0);
            }

            wgpu::Adapter adapter = wgpu::Adapter::Acquire(wAdapter);
            adapter.RequestDevice(
                nullptr,
                [](WGPURequestDeviceStatus status, WGPUDevice wDevice, const char* msg, void* userdata)
                {
                    wgpu::Device device = wgpu::Device::Acquire(wDevice);
                    (*reinterpret_cast<std::function<void(wgpu::Device)>*>(userdata))(device);
                },
                userdata
            );
        },
        (void*)&callback
    );
}

void WebGPURFRender::InitGraphics(wgpu::Surface surface)
{
    SetupSwapChain(surface);
    CreateRenderPipeline();
}

void WebGPURFRender::RenderWrapper(void* userdata)
{
    WebGPURFRender* pInst = static_cast<WebGPURFRender*>(userdata);
    pInst->Render();
}

void WebGPURFRender::Render()
{
    if (!isStartRendering)
    {
        LogToConsole("rendering is not started!");
        return;
    }

    wgpu::RenderPassColorAttachment attachment
    {
        .view = mSwapChain.GetCurrentTextureView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store
    };

    wgpu::RenderPassDescriptor renderPassDesc{
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment,
    };

    mEncoder = mDevice.CreateCommandEncoder();
    mPassEncoder = mEncoder.BeginRenderPass(&renderPassDesc);
    mPassEncoder.SetPipeline(mPipeline);

    size_t ySize = width * height;
    size_t uSize = width / 2 * height / 2;
    size_t yOffset = 0;
    size_t uOffset = ySize;
    size_t vOffset = ySize + uSize;

    uint8_t* yPlaneData = new uint8_t[ySize];
    uint8_t* uPlaneData = new uint8_t[uSize];
    uint8_t* vPlaneData = new uint8_t[uSize];

    memcpy(yPlaneData, yuvBuf.yuvbuf + yOffset, ySize);
    memcpy(uPlaneData, yuvBuf.yuvbuf + uOffset, uSize);
    memcpy(vPlaneData, yuvBuf.yuvbuf + vOffset, uSize);

    wgpu::Extent3D yTexExt
    {
        .width = width,
        .height = height,
        .depthOrArrayLayers = 1
    };

    wgpu::Extent3D uvTexExt
    {
        .width = width / 2,
        .height = height / 2,
        .depthOrArrayLayers = 1
    };

    // the index = 1/2/3 are the texture entries
    wgpu::TextureDescriptor yTexDesc
    {
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = yTexExt,
        .format = wgpu::TextureFormat::R8Unorm,
    };
    wgpu::Texture yTex = mDevice.CreateTexture(&yTexDesc);

    wgpu::ImageCopyTexture yPlaneImgTex
    {
        .texture = yTex,
    };

    wgpu::TextureDataLayout yPlaneDataLayout
    {
        .bytesPerRow = width
    };

    mDevice.GetQueue().WriteTexture(&yPlaneImgTex, yPlaneData, ySize, &yPlaneDataLayout, &yTexExt);

    wgpu::TextureViewDescriptor yTexViewDesc
    {
        .format = wgpu::TextureFormat::R8Unorm,
        .aspect = wgpu::TextureAspect::All
    };
    wgpu::TextureView yTexView = yTex.CreateView(&yTexViewDesc);

    wgpu::TextureDescriptor uTexDesc
    {
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = uvTexExt,
        .format = wgpu::TextureFormat::R8Unorm,
    };
    wgpu::Texture uTex = mDevice.CreateTexture(&uTexDesc);

    wgpu::ImageCopyTexture uPlaneImgTex
    {
        .texture = uTex,
    };

    wgpu::TextureDataLayout uvPlaneDataLayout
    {
        .bytesPerRow = width / 2,
    };
        
    mDevice.GetQueue().WriteTexture(&uPlaneImgTex, uPlaneData, uSize, &uvPlaneDataLayout, &uvTexExt);

    wgpu::TextureViewDescriptor uTexViewDesc
    {
        .format = wgpu::TextureFormat::R8Unorm,
        .aspect = wgpu::TextureAspect::All
    };
    wgpu::TextureView uTexView = uTex.CreateView(&uTexViewDesc);

    wgpu::TextureDescriptor vTexDesc
    {
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = uvTexExt,
        .format = wgpu::TextureFormat::R8Unorm,
    };
    wgpu::Texture vTex = mDevice.CreateTexture(&vTexDesc);

    wgpu::ImageCopyTexture vPlaneImgTex
    {
        .texture = vTex,
    };

    mDevice.GetQueue().WriteTexture(&vPlaneImgTex, vPlaneData, uSize, &uvPlaneDataLayout, &uvTexExt);

    wgpu::TextureViewDescriptor vTexViewDesc
    {
        .format = wgpu::TextureFormat::R8Unorm,
        .aspect = wgpu::TextureAspect::All
    };
    wgpu::TextureView vTexView = vTex.CreateView(&vTexViewDesc);

    // the index = 0 is the sampler entry
    wgpu::BindGroupEntry sampler0Entry
    {
        .binding = 0,
        .sampler = mSampler0,
    };

    wgpu::BindGroupEntry yPlaneTexEntry
    {
        .binding = 1,
        .textureView = yTexView
    };

    wgpu::BindGroupEntry uPlanesTexEntry
    {
        .binding = 2,
        .textureView = uTexView
    };

    wgpu::BindGroupEntry vPlanesTexEntry
    {
        .binding = 3,
        .textureView = vTexView
    };

    wgpu::BufferDescriptor outputBufDesc
    {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        .size = 1000,
        //.mappedAtCreation = true,
    };

    wgpu::BufferDescriptor stagingBufDesc
    {
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
        .size = 1000,
        // .mappedAtCreation = true,
    };

    outputBuffer = mDevice.CreateBuffer(&outputBufDesc);
    stagingBuffer = mDevice.CreateBuffer(&stagingBufDesc);
    wgpu::BindGroupEntry outputBufEntry
    {
        .binding = 4,
        .buffer = outputBuffer,
    };

    std::vector<wgpu::BindGroupEntry> blEntries;
    blEntries.push_back(sampler0Entry);
    blEntries.push_back(yPlaneTexEntry);
    blEntries.push_back(uPlanesTexEntry);
    blEntries.push_back(vPlanesTexEntry);
    blEntries.push_back(outputBufEntry);

    wgpu::BindGroupLayout bgLayout = mPipeline.GetBindGroupLayout(0);
    wgpu::BindGroupDescriptor bgDesc
    {
        .layout = bgLayout,
        .entryCount = 5,
        .entries = blEntries.data()
    };

    wgpu::BindGroup bindGroup = mDevice.CreateBindGroup(&bgDesc);
    mPassEncoder.SetBindGroup(0, bindGroup);
    mPassEncoder.Draw(6);
    mPassEncoder.End();

    //mEncoder.CopyBufferToBuffer(
    //    outputBuffer,
    //    0,
    //    stagingBuffer,
    //    0,
    //    1000
    //);
    //uint32_t* buf = new uint32_t[1000];
    //uint32_t* data = (uint32_t*)stagingBuffer.GetMappedRange(0, 1000);
    //// std::memcpy(buf, data, 1000);
    //stagingBuffer.Unmap();

    wgpu::CommandBuffer cmdBuffer = mEncoder.Finish();
    wgpu::Queue queue = mDevice.GetQueue();
    queue.Submit(1, &cmdBuffer);

    delete[] yPlaneData;
    delete[] uPlaneData;
    delete[] vPlaneData;
}

void WebGPURFRender::SetupSwapChain(wgpu::Surface surface)
{
    wgpu::SwapChainDescriptor scDesc
    {
        .usage = wgpu::TextureUsage::RenderAttachment,
        .format = wgpu::TextureFormat::RGBA8Unorm,
        .width = width,
        .height = height,
        .presentMode = wgpu::PresentMode::Fifo
    };

    mSwapChain = mDevice.CreateSwapChain(surface, &scDesc);
}

void WebGPURFRender::CreateRenderPipeline()
{
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = SHADERS;

    wgpu::ShaderModuleDescriptor shaderModuleDesc{ .nextInChain = &wgslDesc };
    wgpu::ShaderModule shaderModule = mDevice.CreateShaderModule(&shaderModuleDesc);
    wgpu::ColorTargetState colorTargetState = { .format = wgpu::TextureFormat::RGBA8Unorm };
    
    
    wgpu::VertexState vertexState
    {
        .module = shaderModule,
        .entryPoint = "vert_main",
    };
    
    wgpu::FragmentState fragmentState
    {
        .module = shaderModule,
        .entryPoint = "frag_main",
        .targetCount = 1,
        .targets = &colorTargetState
    };

    wgpu::PrimitiveState primitiveState
    {
        .topology = wgpu::PrimitiveTopology::TriangleList,
    };

    wgpu::RenderPipelineDescriptor rpDesc
    {
        .vertex = vertexState,
        .primitive = primitiveState,
        .fragment = &fragmentState,
    };
    mPipeline = mDevice.CreateRenderPipeline(&rpDesc);

    wgpu::SamplerDescriptor samplerDesc{};
    mSampler0 = mDevice.CreateSampler(&samplerDesc);
    mSampler1 = mDevice.CreateSampler(&samplerDesc);
    mSampler2 = mDevice.CreateSampler(&samplerDesc);
}

void WebGPURFRender::LogToConsole(std::string content)
{
#if defined(__EMSCRIPTEN__)
    std::string log = "console.log('" + content + "');";
    emscripten_run_script(log.c_str());
#endif
}