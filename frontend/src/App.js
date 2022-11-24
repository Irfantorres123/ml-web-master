import React, { Component } from "react";
import "antd/dist/antd.css";
import Upload from "./upload";
import ResultTable from "./resultTable";
import logo from './biomiblab_logo.png';
class App extends Component {
  //upload multiple files
  render() {
    return (
      <div>
        <div className="logo">
            <img src={logo} width="200px"/>
        </div>
        <div style={{ margin: "20px",backgroundColor:"#E0AA0F",padding:'0px 10px',width:'800px' }}>
          <h3>Brain MRI UNET</h3>
        </div>
        <div style={{ margin: "20px" }}>
          <h4>Upload Files</h4>
          <Upload />
        </div>
        <div style={{ margin: "20px", }}>
          {" "}
          <h4 style={{
            backgroundColor:"#E0AA0F",width:'800px', padding:'0px 10px'
          }}>Result</h4>
          <ResultTable />
        </div>
      </div>
    );
  }
}
export default App;
