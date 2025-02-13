//
// Created by WildSpirit on 2022/06/23.
//

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H
struct BBoxCenter {
    int xc;
    int yc;
    
    BBoxCenter() {
        xc = -1;
        yc = -1;
    }
    BBoxCenter(int xc, int yc) : xc(xc), yc(yc) {}
};

struct BBox {
    int x;
    int y;
    int w;
    int h;
    
    BBoxCenter getCenter() {
        return BBoxCenter{x + w/2, y + h/2};
    }
    
    BBox() {
        x = y = w = h = 0;
    }
    
    BBox(int _x, int _y, int _w, int _h) {
        x = _x;
        y = _y;
        w = _w;
        h = _h;
    }
    
    template <typename T>
    BBox(T rectBox) {
        x = rectBox.x;
        y = rectBox.y;
        w = rectBox.width;
        h = rectBox.height;
    }
    
};

struct BoxDims
{
    float x;
    float y;
    float w;
    float h;

    BoxDims(float x, float y, float w, float h) : x(x), y(y), w(w), h(h) {}
};

class BoxInfo {
private:
    int trackId;
    int classId;
    float confidenceScore;
    BBox box = BBox();
public:
    BoxInfo() {
        trackId = -1;
        classId = -1;
        confidenceScore = 0;
        box.x = 0;
        box.y = 0;
        box.w = 0;
        box.h = 0;
    }

    BoxInfo(int trackId, int classId, float score, BBox box) {
        this->trackId = trackId;
        this->classId = classId;
        confidenceScore = score;
        this->box = box;
    }


    BoxInfo(int trackId, int classId, float score, int x, int y, int w, int h) {
        this->trackId = trackId;
        this->classId = classId;
        confidenceScore = score;
        box.x = x;
        box.y = y;
        box.w = w;
        box.h = h;
    }

    ~BoxInfo() = default;

    BBox getBox() const {
        return box;
    }
    
    void setBox(BBox bbox) {
        box = bbox;
    }
    
    int getId() {
        return trackId;
    }

    float getConfidence() {
        return confidenceScore;
    }

    int getClassId() const {
        return classId;
    }

    int getLeft() {
        return box.x;
    }

    int getRight() {
        return (box.x + box.w);
    }

    int getTop() {
        return box.y;
    }

    int getBottom() {
        return (box.y + box.h);
    }

    int getWidth() {
        return box.w;
    }

    int getHeight() {
        return box.h;
    }
    
    BBoxCenter getCenter() {
        return BBoxCenter{box.x + box.w/2, box.y + box.h/2};
    }
};

#endif
