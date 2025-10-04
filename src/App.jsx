import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import NavBar from "./components/Navbar";
import Home from "./components/Home";
import WhoWeAre from "./components/WhoWeAre";
function App() {
  return (
    <Router>
      <main className="relative min-h-screen w-full">
        <NavBar />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/who-we-are" element={<WhoWeAre />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
