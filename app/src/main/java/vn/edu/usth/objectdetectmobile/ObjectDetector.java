package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import androidx.annotation.NonNull;
import ai.onnxruntime.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import static java.lang.Math.*;

public class ObjectDetector implements AutoCloseable {
    public static class Detection {
        public final float x1,y1,x2,y2,score,depth; public final int cls;
        public Detection(float x1,float y1,float x2,float y2,float score,int cls){
            this(x1,y1,x2,y2,score,cls,Float.NaN);
        }
        private Detection(float x1,float y1,float x2,float y2,float score,int cls,float depth){
            this.x1=x1; this.y1=y1; this.x2=x2; this.y2=y2; this.score=score; this.cls=cls; this.depth=depth;
        }
        public Detection withDepth(float depthValue){
            return new Detection(x1,y1,x2,y2,score,cls,depthValue);
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int inputW = 640, inputH = 640;
    private final float confThresh = 0.25f, iouThresh = 0.45f;
    private final String inputName;

    public ObjectDetector(@NonNull Context ctx) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        String modelPath = Util.cacheAsset(ctx, "yolov8m_compatible.onnx");
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        // Optional: enable NNAPI on supported devices
        // so.addNnapi();  // See ORT NNAPI EP docs
        session = env.createSession(modelPath, so);
        inputName = session.getInputInfo().keySet().iterator().next();
    }

    public List<Detection> detect(int[] argb, int srcW, int srcH) throws OrtException {
        Letterbox lb = letterbox(argb, srcW, srcH);
        float[] chw = toCHW(lb.rgb, inputW, inputH);
        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw),
                new long[]{1,3,inputH,inputW});

        try (OrtSession.Result out = session.run(Collections.singletonMap(inputName, input))) {
            OnnxValue ov = out.get(0);
            return parse(ov, lb.scale, lb.padX, lb.padY, srcW, srcH);
        }
    }

    // --- preprocessing ---
    private static class Letterbox { int[] rgb; float scale, padX, padY; }
    private Letterbox letterbox(int[] src, int w, int h) {
        float r = Math.min(inputW/(float)w, inputH/(float)h);
        int nw = (int)(w*r), nh = (int)(h*r);
        int dx = (inputW - nw)/2, dy = (inputH - nh)/2;

        int[] dst = new int[inputW*inputH]; // zero-padded
        for (int y=0; y<nh; y++) {
            int sy = Math.min((int)(y/r), h-1);
            for (int x=0; x<nw; x++) {
                int sx = Math.min((int)(x/r), w-1);
                dst[(y+dy)*inputW + (x+dx)] = src[sy*w + sx];
            }
        }
        Letterbox lb = new Letterbox();
        lb.rgb = dst; lb.scale = r; lb.padX = dx; lb.padY = dy;
        return lb;
    }

    private float[] toCHW(int[] rgb, int w, int h) {
        int size = w*h; float[] out = new float[3*size];
        int rI=0, gI=size, bI=2*size;
        for (int i=0;i<size;i++){
            int p = rgb[i];
            out[rI++] = ((p>>16)&0xFF)/255f;
            out[gI++] = ((p>>8)&0xFF)/255f;
            out[bI++] = (p&0xFF)/255f;
        }
        return out;
    }

    // --- parse YOLOv8 output + NMS ---
    private List<Detection> parse(OnnxValue val, float scale, float padX, float padY, int imgW, int imgH) throws OrtException {
        OnnxTensor t = (OnnxTensor) val;
        long[] shape = t.getInfo().getShape(); // expect [1,84,N] or [1,N,84]
        float[] flat = t.getFloatBuffer().array();

        int dim1 = (int)shape[1], dim2 = (int)shape[2];
        boolean colsAreProps = (dim1==84); // [1,84,N]
        int props = colsAreProps ? dim1 : dim2;
        int clsCount = props - 4;
        int N = colsAreProps ? dim2 : dim1;

        List<Detection> dets = new ArrayList<>(N);
        if (colsAreProps) {
            int stride = N; // properties are stored in separate contiguous rows
            for (int i=0;i<N;i++){
                float x = flat[i];
                float y = flat[stride + i];
                float w = flat[2*stride + i];
                float h = flat[3*stride + i];

                int bestC = -1; float bestS = 0f;
                for (int c=0;c<clsCount;c++){
                    float s = flat[(4+c)*stride + i];
                    if (s>bestS){ bestS = s; bestC = c; }
                }
                if (bestS < confThresh) continue;

                float bx = x - w/2f, by = y - h/2f, ex = x + w/2f, ey = y + h/2f;
                float x1 = clamp((bx - padX)/scale, 0, imgW);
                float y1 = clamp((by - padY)/scale, 0, imgH);
                float x2 = clamp((ex - padX)/scale, 0, imgW);
                float y2 = clamp((ey - padY)/scale, 0, imgH);
                dets.add(new Detection(x1,y1,x2,y2,bestS,bestC));
            }
        } else {
            for (int i=0;i<N;i++){
                int base = i*props;
                float x = flat[base+0], y = flat[base+1],
                        w = flat[base+2], h = flat[base+3];

                int bestC = -1; float bestS = 0f;
                for (int c=0;c<clsCount;c++){
                    float s = flat[base+4+c];
                    if (s>bestS){ bestS = s; bestC = c; }
                }
                if (bestS < confThresh) continue;

                float bx = x - w/2f, by = y - h/2f, ex = x + w/2f, ey = y + h/2f;
                float x1 = clamp((bx - padX)/scale, 0, imgW);
                float y1 = clamp((by - padY)/scale, 0, imgH);
                float x2 = clamp((ex - padX)/scale, 0, imgW);
                float y2 = clamp((ey - padY)/scale, 0, imgH);
                dets.add(new Detection(x1,y1,x2,y2,bestS,bestC));
            }
        }
        return nms(dets, iouThresh);
    }

    private static float clamp(float v, int lo, int hi){ return Math.max(lo, Math.min(hi, v)); }

    private static float iou(Detection A, Detection B){
        float ix1 = max(A.x1,B.x1), iy1 = max(A.y1,B.y1);
        float ix2 = min(A.x2,B.x2), iy2 = min(A.y2,B.y2);
        float iw = max(0f, ix2-ix1), ih = max(0f, iy2-iy1);
        float inter = iw*ih;
        float a = (A.x2-A.x1)*(A.y2-A.y1);
        float b = (B.x2-B.x1)*(B.y2-B.y1);
        return inter/(a+b-inter+1e-6f);
    }

    private static List<Detection> nms(List<Detection> in, float iouTh){
        ArrayList<Detection> dets = new ArrayList<>(in);
        dets.sort((d1,d2)-> Float.compare(d2.score, d1.score));
        List<Detection> keep = new ArrayList<>();
        while(!dets.isEmpty()){
            Detection a = dets.remove(0);
            keep.add(a);
            dets.removeIf(b -> b.cls==a.cls && iou(a,b) > iouTh);
        }
        return keep;
    }

    @Override public void close() throws Exception {
        session.close();
    }

    // Utility to read asset fully
    static class Util {
        static byte[] readAllBytes(android.content.res.AssetManager am, String name){
            try(java.io.InputStream is = am.open(name);
                java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream()){
                byte[] buf = new byte[1<<16]; int r;
                while ((r=is.read(buf))!=-1) bos.write(buf,0,r);
                return bos.toByteArray();
            } catch (Exception e){ throw new RuntimeException(e); }
        }

        static String cacheAsset(Context ctx, String assetName){
            File dir = new File(ctx.getFilesDir(), "models");
            if (!dir.exists()) dir.mkdirs();
            File out = new File(dir, assetName);
            if (out.exists() && out.length() > 0) return out.getAbsolutePath();
            try (InputStream is = ctx.getAssets().open(assetName);
                 FileOutputStream fos = new FileOutputStream(out)) {
                byte[] buf = new byte[1<<16]; int r;
                while ((r = is.read(buf)) != -1) fos.write(buf, 0, r);
                fos.flush();
                return out.getAbsolutePath();
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }
    }
}
