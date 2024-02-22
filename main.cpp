#include <WebGPURFRender.h>
#include <include/emscripten.h>

WebGPURFRender *pRender;

extern "C" void ReceiveBuffer(uint8_t* ybuf, size_t ylen, uint8_t* ubuf, size_t ulen, uint8_t* vbuf, size_t vlen)
{
	if (pRender) 
	{
		pRender->OnBufferReceived(ybuf, ylen, ubuf, ulen, vbuf, vlen);
	}
}

extern "C" void ReceiveYUVBuffer(uint8_t* yuvbuf, size_t len)
{
	if (pRender) 
	{
		pRender->OnYUVBufferReceived(yuvbuf, len);
	}
}

extern "C" void StartRendering()
{
	if (pRender)
	{
		pRender->StartRendering();
	}
}

int main()
{
	emscripten_run_script(" console.log('main is called!'); ");
	pRender = new WebGPURFRender();
	pRender->Prepare();
	return 0;
}