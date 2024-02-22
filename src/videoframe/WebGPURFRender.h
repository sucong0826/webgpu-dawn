#include <webgpu/webgpu_cpp.h>
#include <functional>
#include <vector>
#include <utility>
#include <array>
#include <initializer_list>

struct CacheBuf
{
	uint8_t* ybuf;
	size_t ylen;
	uint8_t* ubuf;
	size_t ulen;
	uint8_t* vbuf;
	size_t vlen;
};

struct YUVBuf
{
	uint8_t* yuvbuf;
	size_t len;
	size_t ylen;
	size_t ulen;
	size_t vlen;
};

struct ColorSpaceConversionInfo 
{
	std::array<float, 12> yuvToRgbConversionMatrix;
	std::array<float, 9> gamutConversionMatrix;
	std::array<float, 7> srcTransferFunctionParameters;
	std::array<float, 7> dstTransferFunctionParameters;
};

struct CallbackData {
	wgpu::Device device;
	wgpu::Buffer buffer;
};

class WebGPURFRender 
{
public:
	WebGPURFRender();
	void Prepare();
	void StartRendering();
	void OnBufferReceived(uint8_t* ybuf, size_t ylen, uint8_t* ubuf, size_t ulen, uint8_t* vbuf, size_t vlen);
	void OnYUVBufferReceived(uint8_t* yuvbuf, size_t len);
	static void RenderWrapper(void* userdata);

private:
	virtual ~WebGPURFRender();
	// void GetDevice(void (*callback)(wgpu::Device));
	void GetDevice(std::function<void(wgpu::Device)> callback);
	void Start();
	void InitGraphics(wgpu::Surface surface);
	void SetupSwapChain(wgpu::Surface surface);
	void CreateRenderPipeline();
	void Render();
	void RenderYUVTexture();
	void CreateExternalTexture();
	void LogToConsole(std::string content);
};

