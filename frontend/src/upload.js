import React, { useRef, useState } from "react";
import "antd/dist/antd.css";
import "./index.css";
import { UploadOutlined, DeleteOutlined } from "@ant-design/icons";
import { Button, Upload, Row, Col } from "antd";
import axios from "axios";
import { updateFileList } from "antd/lib/upload/utils";

const App = () => {
  const image = useRef(null);
  // upload props
  const props = {
    action: "api/upload",
    multiple: false,
    onChange({ file, fileList, event }) {
      file.status = "done";
    },
  };
  // clear uploaded files
  const handleClear = async () => {
    await axios.get("/api/clear").then((res) => {
      console.log(res.status);
      if (res.status === 200) {
      }
    });
  };
  async function readURL(input) {
    return new Promise((resolve, reject) => {
      if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
          resolve(e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
      }
    });
  }

  return (
    <div>
      <Row>
        <Col span={6}>
          <input
            type="file"
            name="filename"
            onChange={async (e) => {
              const formData = new FormData();
              formData.append("file", e.target.files[0]);
              fetch("/api/upload", {
                method: "POST",

                body: formData,
              });
            }}
          />
        </Col>
        <Col span={4}>
          <Button icon={<DeleteOutlined />} onClick={handleClear}>
            Clear all
          </Button>
        </Col>
      </Row>
      <Row>
        <Col span={6} id="images"></Col>
      </Row>
    </div>
  );
};

export default App;
