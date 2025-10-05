import { useNavigate } from "react-router-dom";
import AnimatedTitle from "./AnimatedTitle";
import Button from "./Button";

const ImageClipBox = ({ src, clipClass }) => (
  <div className={clipClass}>
    <img src={src} />
  </div>
);

const Footer = () => {
  const navigate = useNavigate();

  const handleContactClick = () => {
    navigate("/who-we-are");
  };
  return (
    <div className="my-10 min-h-96 w-full overflow-hidden px-10">
      <div className="relative rounded-lg bg-black py-24 text-[#dfdff2] sm:overflow-hidden">
        <div className="absolute -left-20 top-0 hidden h-full w-72 overflow-hidden sm:block lg:left-20 lg:w-96">
          <ImageClipBox src="/img/lepng.avif" clipClass="contact-clip-path-1" />
        </div>

        <div className="absolute -top-40 left-20 w-60 sm:top-1/2 md:left-auto md:right-10 lg:top-20 lg:w-80">
          <ImageClipBox
            src="/img/curry.png"
            clipClass="absolute md:scale-125"
          />
          <ImageClipBox
            src="/img/curry.png"
            clipClass="sword-man-clip-path md:scale-125"
          />
        </div>

        <div className="flex flex-col items-center text-center">
          <p className="mb-10 font-general text-[10px] uppercase">
            Project created for NASA SPACE APPS 2025 CHICAGO
          </p>

          <AnimatedTitle
            title="Team Naasa"
            className="special-font !md:text-[6.2rem] w-full font-zentry !text-5xl !font-black !leading-[.9]"
          />

          <Button
            title="contact us"
            containerClass="mt-10 cursor-pointer"
            onClick={handleContactClick}
          />
        </div>
      </div>
    </div>
  );
};

export default Footer;
