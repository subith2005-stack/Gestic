import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"

function App(){

const videoRef = useRef(null)
const canvasRef = useRef(null)

const [gesture,setGesture] = useState("None")
const [confidence,setConfidence] = useState(0)
const [speaking,setSpeaking] = useState(false)

useEffect(()=>{

startCamera()

const interval = setInterval(()=>{
detectGesture()
},1200)

return ()=>clearInterval(interval)

},[])

const startCamera = async ()=>{

const stream = await navigator.mediaDevices.getUserMedia({
video:true
})

videoRef.current.srcObject = stream
}

const detectGesture = async ()=>{

const video = videoRef.current
const canvas = canvasRef.current

if(!video || !canvas) return

const ctx = canvas.getContext("2d")

canvas.width = 128
canvas.height = 128

ctx.drawImage(video,0,0,128,128)

canvas.toBlob(async(blob)=>{

const formData = new FormData()
formData.append("file",blob,"frame.jpg")

try{

const res = await fetch("http://127.0.0.1:8000/predict",{
method:"POST",
body:formData
})

const data = await res.json()

setGesture(data.gesture)
setConfidence((data.confidence*100).toFixed(2))

}catch(err){
console.log(err)
}

},"image/jpeg")
}

const speakGesture = ()=>{

if(!gesture || gesture==="None") return

const utter = new SpeechSynthesisUtterance(gesture)

setSpeaking(true)

utter.onend = ()=>{
setSpeaking(false)
}

speechSynthesis.speak(utter)
}

return(

<div className="app">

{/* BACKGROUND BLOBS */}

<motion.div
className="blob blob1"
animate={{x:[0,120,-80,0],y:[0,-80,60,0]}}
transition={{duration:25,repeat:Infinity,repeatType:"mirror",ease:"easeInOut"}}
/>

<motion.div
className="blob blob2"
animate={{x:[0,-100,80,0],y:[0,80,-60,0]}}
transition={{duration:30,repeat:Infinity,repeatType:"mirror",ease:"easeInOut"}}
/>

<motion.div
className="blob blob3"
animate={{x:[0,60,-60,0],y:[0,-50,90,0]}}
transition={{duration:35,repeat:Infinity,repeatType:"mirror",ease:"easeInOut"}}
/>

<div className="layout">

{/* CAMERA PANEL */}

<div className="cameraPanel">

<h1>GESTIC</h1>

<video
ref={videoRef}
autoPlay
playsInline
className="camera"
/>

<canvas ref={canvasRef} style={{display:"none"}}/>

</div>

{/* INFO PANEL */}

<div className="infoPanel">

<motion.div
key={gesture}
initial={{scale:0.8,opacity:0}}
animate={{scale:1,opacity:1}}
transition={{duration:0.4}}
className="gestureCard"

>

<h2>{gesture}</h2>
<p>{confidence}% confidence</p>

</motion.div>

<button onClick={speakGesture}>
Speak Gesture
</button>

{/* AUDIO WAVE */}

{speaking && (

<div className="wave">

<span></span> <span></span> <span></span> <span></span> <span></span>

</div>

)}

</div>

</div>

</div>

)
}

export default App

