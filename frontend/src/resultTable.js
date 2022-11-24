import "antd/dist/antd.css";
import "./index.css";

import axios from "axios";
import {
  Button,
  Form,
  Input,
  Popconfirm,
  Table,
  Space,
  Image,
  notification,
} from "antd";
import React, { useContext, useEffect, useRef, useState } from "react";
import Excel from "exceljs";
import * as FileSaver from "file-saver";

const App = () => {
  const [images, setImages] = useState(null);
  const [actualImage, setActualImage] = useState(null);
  const [probability, setProbability] = useState(null);
  // get result
  const handleUpdate = async () => {
    await axios
      .get("/api/result-prediction")
      .then(async (res) => {
        console.log(res);
        let { original, prediction, prediction_t, probability, cropped_image } =
          res.data;
        setImages([prediction, prediction_t, cropped_image]);
        setActualImage(original);
        setProbability(probability);
      })
      .catch(async (res) => {
        console.log(res);
        if (res.status === 415) {
          notification.open({
            message: "Invalid Input!",
            description: "Please use Clear all to reset your input",
          });
        }
        if (res.status === 404) {
          setImages([]);
          notification.open({
            message: "Input not found!",
            description: "Please upload your input files",
          });
        }
      });
  };

  // get result and export excel
  const handleDownload = async () => {
    await axios
      .get("/api/result-scoliosis")
      .then((res) => {
        console.log("SUCCESSS");
        const workbook = new Excel.Workbook();
        workbook.creator = "Web";
        workbook.lastModifiedBy = "Web";
        workbook.created = new Date();
        workbook.modified = new Date();
        workbook.lastPrinted = new Date();
        const worksheet = workbook.addWorksheet("Sheet 1");
        worksheet.columns = [
          { header: "name", key: "name", width: 10 },
          { header: "PT", key: "PT", width: 10 },
          { header: "MT", key: "MT", width: 10 },
          { header: "TL", key: "TL", width: 10 },
        ];
        worksheet.addRows(res.data);
        console.log(workbook.xlsx);
        workbook.xlsx.writeBuffer().then((data) => {
          const blob = new Blob([data], {
            type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8",
          });
          // Given name
          FileSaver.saveAs(blob, "download.xlsx");
        });
      })
      .catch((err) => {
        console.log(err.response.status);
        if (err.response.status === 415) {
          notification.open({
            message: "Invalid Input!",
            description: "Please use Clear all to reset your input",
          });
        }
        if (err.response.status === 404) {
          notification.open({
            message: "Input not found!",
            description: "Please upload your input files",
          });
        }
      });
  };

  return (
    <div>
      <Space>
        <Button
          onClick={handleUpdate}
          type="primary"
          style={{
            marginBottom: 16,
          }}
        >
          get result
        </Button>
        <Button
          onClick={handleDownload}
          type="primary"
          style={{
            marginBottom: 16,
          }}
        >
          download result
        </Button>
      </Space>
      <div>
        <h1>Probability of tumor</h1>
        <h2>{probability} %</h2>
      </div>
      <div style={{ display: "flex" }}>
        {actualImage && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              marginRight: "4rem",
            }}
          >
            <h2>Original Image</h2>

            <div display="flex">
              <img
                src={`data:image/png;base64,${actualImage}`}
                width="400px"
                alt="original"
              />
              <br />
            </div>
          </div>
        )}
        {images && (
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex" }}>
              {images.map((image, index) => (
                <div
                  key={image}
                  style={{
                    marginRight: "1rem",
                  }}
                >
                  <h2>{index === 0 && "Mask with threshold"}</h2>
                  <h2>{index === 1 && "Mask"}</h2>
                  <h2>{index === 2 && "Tumor area expanded"}</h2>
                  <img
                    src={`data:image/png;base64,${image}`}
                    width="400px"
                    alt=""
                    style={{
                      marginRight: "1px",
                      marginBottom: "1px",
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
