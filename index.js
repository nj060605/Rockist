// const multer = require('multer');
// const path = require('path');
// const fs = require('fs');
import multer from "multer";
import fs from 'fs'
import path from "path";
import e from "express";
import fileUpload from "express-fileupload";
import bodyParser from "body-parser";
import { exec } from "child_process";
import cors from "cors"

const app = e()
const port = 8000

app.use(cors({
  origin: "http://localhost:3000",
  methods: ["GET", "POST", "PUT", "PATCH", "DELETE"], 
  credentials: true,
}))

app.use(e.static('public'));
app.use(e.urlencoded({extended:false}))
app.use(e.json())
app.use(fileUpload());

app.use(
    bodyParser.urlencoded({
      extended: true,
    })
  );

app.post('/', (req, res) => {
  console.log(req.body)
  res.send('Hello World!')
})


app.post("/segment", (req, res) => {
  const img = req.files['image'];
  const imageData = img['data'];
  const filename = img['name'];
    if (!img || !filename) {
        return res.status(400).json({ message: 'Image data and filename are required' });
    }
    const filePath = path.join('./uploads/', filename);
    if (!fs.existsSync('uploads')) {
        fs.mkdirSync('uploads');
    }
    fs.writeFile(filePath, imageData, (err) => {
        if (err) {
            return res.status(500).json({ message: 'Error saving image', error: err });
        }
        // res.status(200).json({ message: 'Image uploaded successfully', filePath });
    });
  const { imagePath, removeBackground, areaScale } = req.body;
  const command = `python grain_segmentation.py ./uploads/${filename} ${true} ${0.5}`;
  exec(command, (error, stdout, stderr) => {
      if (error) {
          console.error(`Error: ${error.message}`);
          return res.status(500).json({ error: "Error running Python script" });
      }
      if (stderr) {
          console.error(`Stderr: ${stderr}`);
          return res.status(500).json({ error: "Python script error" });
      }
      try {
          const output = JSON.parse(stdout);
          res.json({ data: output });
      } catch (parseError) {
          console.error("Parse Error:", parseError);
          res.status(500).json({ error: "Failed to parse Python output" });
      }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

